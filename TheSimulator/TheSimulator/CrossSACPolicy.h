#pragma once

#include <torch/torch.h>
#include <tuple>
#include <unistd.h>
#include <cstdio>
#include <vector>

struct CrossSACPolicyConfig {
    int low_freq_feature_dim{5};
    int low_freq_seq_len{60};
    int num_assets{4};
    int conv_out_channels{64};
    int conv_kernel_size{1};
    int pool_kernel_size{1};
    int lstm_hidden_size{128};
    int attention_num_heads{8};
    double attention_dropout{0.1};
    int market_reduce_dim{256};
    int weight_fc_dim{256};
    int hidden_dim{256};
    double mlp_dropout{0.1};
    double cnn_dropout{0.2};
    double min_concentration{0.05};
    double sampling_eps{0.01};
    unsigned long long sampling_seed{0ull};
};

struct MarketCNNLSTMEncoderImpl : torch::nn::Module {
    MarketCNNLSTMEncoderImpl(int num_assets,
                             int seq_len,
                             int feature_dim,
                             int conv_out_channels,
                             int conv_kernel_size,
                             int pool_kernel_size,
                             int lstm_hidden_size,
                             double cnn_dropout)
        : m_num_assets(num_assets)
        , m_seq_len(seq_len)
        , m_feature_dim(feature_dim)
        , conv(torch::nn::Conv1dOptions(feature_dim, conv_out_channels, conv_kernel_size).padding(0))
        , conv_act(nullptr)
        , cnn_do(torch::nn::DropoutOptions().p(cnn_dropout))
        , pool(torch::nn::MaxPool1dOptions(pool_kernel_size))
        , lstm(nullptr) {
        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(conv_out_channels, lstm_hidden_size).batch_first(true)));
        register_module("conv", conv);
        conv_act = register_module("conv_act", torch::nn::ReLU());
        register_module("cnn_do", cnn_do);
        register_module("pool", pool);
    }

    torch::Tensor forward(const torch::Tensor& state) {
        auto B = state.size(0);
        auto N = state.size(1);
        auto T = state.size(2);
        auto F = state.size(3);
        (void)T; (void)F;
        auto x = state.reshape({B * N, m_seq_len, m_feature_dim});               // [B*N, T, F]
        x = x.permute({0, 2, 1});                                               // [B*N, F, T]
        x = conv->forward(x);                                                   // [B*N, C, T']
        x = conv_act->forward(x);                                               // ReLU
        x = cnn_do->forward(x);                                                 // Dropout
        x = pool->forward(x);                                                   // [B*N, C, T'']
        x = x.permute({0, 2, 1});                                               // [B*N, T'', C]
        auto lstm_out = std::get<0>(lstm->forward(x));                          // [B*N, T'', H]
        auto last = lstm_out.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // [B*N, H]
        last = last.reshape({B, N, last.size(1)});                              // [B, N, H]
        return last;
    }

    int m_num_assets;
    int m_seq_len;
    int m_feature_dim;
    torch::nn::Conv1d conv{nullptr};
    torch::nn::ReLU conv_act{nullptr};
    torch::nn::Dropout cnn_do{nullptr};
    torch::nn::MaxPool1d pool{nullptr};
    torch::nn::LSTM lstm{nullptr};
};
TORCH_MODULE(MarketCNNLSTMEncoder);

struct CrossSACPolicyImpl : torch::nn::Module {
    explicit CrossSACPolicyImpl(const CrossSACPolicyConfig& cfg)
        : m_cfg(cfg)
        , encoder(nullptr)
        , mha(nullptr)
        , policy_head(nullptr)
        , alpha_layer(nullptr) {
        encoder = register_module("encoder", MarketCNNLSTMEncoder(
            cfg.num_assets,
            cfg.low_freq_seq_len,
            cfg.low_freq_feature_dim,
            cfg.conv_out_channels,
            cfg.conv_kernel_size,
            cfg.pool_kernel_size,
            cfg.lstm_hidden_size,
            cfg.cnn_dropout
        ));

        mha = register_module("mha", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(cfg.lstm_hidden_size, cfg.attention_num_heads).dropout(cfg.attention_dropout)));
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({(long long)cfg.lstm_hidden_size})));

        market_reducer = register_module("market_reducer", torch::nn::Sequential(
            torch::nn::Linear(cfg.num_assets * cfg.lstm_hidden_size, cfg.market_reduce_dim),
            torch::nn::ReLU()
        ));
        weight_encoder = register_module("weight_encoder", torch::nn::Sequential(
            torch::nn::Linear(cfg.num_assets, cfg.weight_fc_dim),
            torch::nn::ReLU()
        ));

        policy_head = register_module("policy_head", torch::nn::Sequential(
            torch::nn::Linear(cfg.market_reduce_dim + cfg.weight_fc_dim, cfg.hidden_dim),
            torch::nn::ReLU(),
            torch::nn::Dropout(cfg.mlp_dropout),
            torch::nn::Linear(cfg.hidden_dim, cfg.hidden_dim),
            torch::nn::ReLU(),
            torch::nn::Dropout(cfg.mlp_dropout)
        ));

        alpha_layer = register_module("alpha_layer", torch::nn::Linear(cfg.hidden_dim, cfg.num_assets));

        m_rng_inited = false;
        if (m_cfg.sampling_seed != 0ull) {
            m_rng.seed(m_cfg.sampling_seed);
            m_rng_inited = true;
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state, const torch::Tensor& weights, bool deterministic = true) {
        auto B = state.size(0);
        auto N = state.size(1);
        (void)N;
        auto per_asset = encoder->forward(state); // [B, N, H]

        auto seq = per_asset.permute({1, 0, 2}); // [N, B, H]
        auto attn_out = std::get<0>(mha->forward(seq, seq, seq));
        auto attn_bn = attn_out.permute({1, 0, 2}); // [B, N, H]

        auto per_asset_post = layer_norm->forward(attn_bn + per_asset); // [B, N, H]

        auto market_flat = per_asset_post.reshape({B, (long long)N * m_cfg.lstm_hidden_size}); // [B, N*H]
        auto market_vec = market_reducer->forward(market_flat); // [B, market_reduce_dim]

        auto weight_vec = weight_encoder->forward(weights); // [B, weight_fc_dim]

        auto fused = torch::cat({market_vec, weight_vec}, 1); // [B, market_reduce_dim+weight_fc_dim]
        auto h = policy_head->forward(fused);

        auto alpha_unconstrained = alpha_layer->forward(h);                       // [B, N]
        auto concentration = torch::nn::functional::softplus(
            alpha_unconstrained,
            torch::nn::functional::SoftplusFuncOptions()
        ) + (float)m_cfg.min_concentration;                                       // [B, N]

        torch::Tensor action;
        if (deterministic) {
            action = concentration / concentration.sum(/*dim=*/1, /*keepdim=*/true);
        } else {
            auto conc_cpu = concentration.to(torch::kCPU).contiguous();
            action = torch::zeros_like(concentration).to(torch::kCPU);
            if (!m_rng_inited) {
                if (m_cfg.sampling_seed != 0ull) m_rng.seed(m_cfg.sampling_seed);
                else m_rng.seed(std::random_device{}());
                m_rng_inited = true;
            }
            for (int64_t b = 0; b < conc_cpu.size(0); ++b) {
                std::vector<double> g((size_t)conc_cpu.size(1), 0.0);
                double sumg = 0.0;
                for (int64_t i = 0; i < conc_cpu.size(1); ++i) {
                    double a_i = std::max(1e-12, (double)conc_cpu[b][i].item<float>());
                    std::gamma_distribution<double> gamma_dist(a_i, 1.0);
                    double gi = gamma_dist(m_rng);
                    g[(size_t)i] = gi; sumg += gi;
                }
                if (!(sumg > 0.0)) { sumg = (double)conc_cpu.size(1); for (auto& v : g) v = 1.0; }
                double eps = (m_cfg.sampling_eps > 0.0 ? m_cfg.sampling_eps : 1e-6);
                for (int64_t i = 0; i < conc_cpu.size(1); ++i) {
                    double ai = g[(size_t)i] / sumg;
                    ai = (1.0 - eps) * ai + eps * (1.0 / (double)conc_cpu.size(1));
                    action[b][i] = (float)ai;
                }
            }
            action = action.to(concentration.device());
        }
        return {concentration, action};
    }

    bool load_from(const std::string& path) {
        try {
            torch::serialize::InputArchive archive;
            archive.load_from(path);
            this->load(archive);
            return true;
        } catch (const std::exception& ex) {
            std::cerr << "[CrossSACPolicy] load_from failed: " << ex.what() << " path=" << path << std::endl;
            return false;
        } catch (...) {
            std::cerr << "[CrossSACPolicy] load_from failed: unknown error path=" << path << std::endl;
            return false;
        }
    }

    bool load_from_bytes(const std::vector<char>& bytes) {
        try {
            char tmpl[] = "/tmp/desmar_actor_params_XXXXXX.pt";
            int fd = mkstemps(tmpl, 3);
            if (fd == -1) {
                std::cerr << "[CrossSACPolicy] mkstemps failed" << std::endl;
                return false;
            }
            FILE* f = fdopen(fd, "wb");
            if (!f) { close(fd); std::cerr << "[CrossSACPolicy] fdopen failed" << std::endl; return false; }
            size_t wrote = fwrite(bytes.data(), 1, bytes.size(), f);
            fclose(f);
            if (wrote != bytes.size()) {
                std::cerr << "[CrossSACPolicy] fwrite short write: wrote=" << wrote << " expect=" << bytes.size() << std::endl;
                unlink(tmpl);
                return false;
            }
            bool ok = load_from(std::string(tmpl));
            unlink(tmpl);
            return ok;
        } catch (const std::exception& ex) {
            std::cerr << "[CrossSACPolicy] load_from_bytes failed: " << ex.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "[CrossSACPolicy] load_from_bytes failed: unknown error" << std::endl;
            return false;
        }
    }

    CrossSACPolicyConfig m_cfg;
    MarketCNNLSTMEncoder encoder;
    torch::nn::MultiheadAttention mha{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};
    torch::nn::Sequential market_reducer{nullptr};
    torch::nn::Sequential weight_encoder{nullptr};
    torch::nn::Sequential policy_head{nullptr};
    torch::nn::Linear alpha_layer{nullptr};
    // RNG for Dirichlet sampling
    std::mt19937_64 m_rng;
    bool m_rng_inited{false};
};
TORCH_MODULE(CrossSACPolicy);