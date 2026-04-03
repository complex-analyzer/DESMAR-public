#pragma once

#include <torch/torch.h>
#include <tuple>
#include <vector>

struct CrossBDQPolicyConfig {
    int lob_depth{5};
    int lob_history_len{60};
    int lob_feature_dim{0};
    int ratio_branches{6};
    int lstm_hidden_size{128};
    int trade_hidden{128};
    int fused_dim{256};
    double mlp_dropout{0.1};
    int cnn_pair_filters{16};
    double cnn_leaky_relu_neg_slope{0.01};
    int temporal_kernel{2};
    int temporal_repeats{2};
    bool inception_enabled{true};
    int inception_out_channels{32};
    int inception_pool_time_kernel{3};
    double lstm_input_dropout{0.2};
    double lstm_output_dropout{0.0};
};

// --- LOBEncoder ---
struct BDQ_LOBEncoderImpl : torch::nn::Module {
    explicit BDQ_LOBEncoderImpl(const CrossBDQPolicyConfig& cfg)
        : m_cfg(cfg)
        , conv12_a(nullptr)
        , conv12_b(nullptr)
        , conv_levels(nullptr)
        , lstm(nullptr)
        , input_do(torch::nn::DropoutOptions().p(cfg.lstm_input_dropout))
        , output_do(torch::nn::DropoutOptions().p(cfg.lstm_output_dropout))
        , act(register_module("act", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(cfg.cnn_leaky_relu_neg_slope)))) {
        const int C = cfg.cnn_pair_filters;
        // 1x2 stride(1,2)
        conv12_a = register_module("conv12_a", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, C, {1, 2}).stride({1, 2}).padding({0, 0}).bias(true)));
        conv12_b = register_module("conv12_b", torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, {1, 2}).stride({1, 2}).padding({0, 0}).bias(true)));

        // temporal conv groups
        auto pad_t = std::max(0, (cfg.temporal_kernel - 1) / 2);
        auto make_temporal_group = [&](const std::string& name) {
            torch::nn::ModuleList group;
            for (int i = 0; i < std::max(1, cfg.temporal_repeats); ++i) {
                auto m = torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, {cfg.temporal_kernel, 1}).stride({1, 1}).padding({pad_t, 0}).bias(true));
                group->push_back(m);
            }
            register_module(name, group);
            return group;
        };
        convs_t_after_a = make_temporal_group("convs_t_after_a");
        convs_t_after_b = make_temporal_group("convs_t_after_b");

        // 1xD across depth
        conv_levels = register_module("conv_levels", torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, {1, cfg.lob_depth}).stride({1, 1}).padding({0, 0}).bias(true)));
        convs_t_after_levels = make_temporal_group("convs_t_after_levels");

        // Inception on time dim (width=1)
        inception_enabled = cfg.inception_enabled;
        if (inception_enabled) {
            int out_each = std::max(1, cfg.inception_out_channels / 4);
            incp_1x1_a = register_module("incp_1x1_a", torch::nn::Conv2d(torch::nn::Conv2dOptions(C, out_each, {1, 1})));
            incp_1x1_b3 = register_module("incp_1x1_b3", torch::nn::Conv2d(torch::nn::Conv2dOptions(C, out_each, {1, 1})));
            incp_1x1_b5 = register_module("incp_1x1_b5", torch::nn::Conv2d(torch::nn::Conv2dOptions(C, out_each, {1, 1})));
            incp_3x1 = register_module("incp_3x1", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_each, out_each, {3, 1}).padding({1, 0})));
            incp_5x1 = register_module("incp_5x1", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_each, out_each, {5, 1}).padding({2, 0})));
            incp_pool = register_module("incp_pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({cfg.inception_pool_time_kernel, 1}).stride({1, 1}).padding({cfg.inception_pool_time_kernel / 2, 0})));
            incp_pool_1x1 = register_module("incp_pool_1x1", torch::nn::Conv2d(torch::nn::Conv2dOptions(C, out_each, {1, 1})));
            incp_act = register_module("incp_act", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(cfg.cnn_leaky_relu_neg_slope)));
            incp_out_channels = out_each * 4;
            lstm_in_channels = incp_out_channels;
        } else {
            lstm_in_channels = C;
        }

        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(lstm_in_channels, cfg.lstm_hidden_size).batch_first(true)));
    }

    torch::Tensor forward(const torch::Tensor& lob_seq) {
        // lob_seq: [B, K, 4D]
        auto x = lob_seq.unsqueeze(1);                               // [B,1,K,F]
        x = act->forward(conv12_a->forward(x));                      // [B,C,K,2D]
        for (auto& m : *convs_t_after_a) x = act->forward(m->as<torch::nn::Conv2d>()->forward(x));
        x = act->forward(conv12_b->forward(x));                      // [B,C,K,D]
        for (auto& m : *convs_t_after_b) x = act->forward(m->as<torch::nn::Conv2d>()->forward(x));
        x = act->forward(conv_levels->forward(x));                   // [B,C,K,1]
        for (auto& m : *convs_t_after_levels) x = act->forward(m->as<torch::nn::Conv2d>()->forward(x));
        x = x.squeeze(-1);                                           // [B,C,K]
        x = x.transpose(1, 2).contiguous();                          // [B,K,C]

        if (inception_enabled) {
            auto xi = x.transpose(1, 2).unsqueeze(-1);               // [B,C,K,1]
            auto b1 = incp_act->forward(incp_1x1_a->forward(xi));
            auto b2 = incp_act->forward(incp_3x1->forward(incp_1x1_b3->forward(xi)));
            auto b3 = incp_act->forward(incp_5x1->forward(incp_1x1_b5->forward(xi)));
            auto b4 = incp_act->forward(incp_pool_1x1->forward(incp_pool->forward(xi)));
            auto cat = torch::cat({b1, b2, b3, b4}, 1);              // [B,C',K,1]
            x = cat.squeeze(-1).transpose(1, 2).contiguous();        // [B,K,C']
        }
        x = input_do->forward(x);
        auto out = std::get<0>(lstm->forward(x));                    // [B,K,H]
        auto feat = out.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // [B,H]
        return output_do->forward(feat);
    }

    CrossBDQPolicyConfig m_cfg;
    torch::nn::Conv2d conv12_a{nullptr}, conv12_b{nullptr}, conv_levels{nullptr};
    torch::nn::ModuleList convs_t_after_a{nullptr}, convs_t_after_b{nullptr}, convs_t_after_levels{nullptr};
    bool inception_enabled{true};
    int incp_out_channels{0};
    int lstm_in_channels{0};
    torch::nn::Conv2d incp_1x1_a{nullptr}, incp_1x1_b3{nullptr}, incp_1x1_b5{nullptr};
    torch::nn::Conv2d incp_3x1{nullptr}, incp_5x1{nullptr};
    torch::nn::MaxPool2d incp_pool{nullptr};
    torch::nn::Conv2d incp_pool_1x1{nullptr};
    torch::nn::LeakyReLU incp_act{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Dropout input_do{nullptr}, output_do{nullptr};
    torch::nn::LeakyReLU act{nullptr};
};
TORCH_MODULE(BDQ_LOBEncoder);

// --- TradingFeatureEncoder ---
struct BDQ_TradingFeatureEncoderImpl : torch::nn::Module {
    explicit BDQ_TradingFeatureEncoderImpl(int hidden_dim)
        : net(register_module("net", torch::nn::Sequential(
              torch::nn::Linear(4, hidden_dim),
              torch::nn::ReLU(),
              torch::nn::Linear(hidden_dim, hidden_dim),
              torch::nn::ReLU()))) {}
    torch::Tensor forward(const torch::Tensor& trade_vec) { return net->forward(trade_vec); }
    torch::nn::Sequential net{nullptr};
};
TORCH_MODULE(BDQ_TradingFeatureEncoder);

// --- SharedStateFusion ---
struct BDQ_SharedStateFusionImpl : torch::nn::Module {
    BDQ_SharedStateFusionImpl(int lob_hidden, int trade_hidden, int fused_dim, double mlp_dropout)
        : fc(register_module("fc", torch::nn::Sequential(
              torch::nn::Linear(lob_hidden + trade_hidden, fused_dim),
              torch::nn::ReLU(),
              torch::nn::Dropout(mlp_dropout),
              torch::nn::Linear(fused_dim, fused_dim),
              torch::nn::ReLU(),
              torch::nn::Dropout(mlp_dropout)))) {}
    torch::Tensor forward(const torch::Tensor& lob_feat, const torch::Tensor& trade_feat) {
        auto fused = torch::cat({lob_feat, trade_feat}, -1);
        return fc->forward(fused);
    }
    torch::nn::Sequential fc{nullptr};
};
TORCH_MODULE(BDQ_SharedStateFusion);

// --- BDQCore ---
struct CrossBDQPolicyImpl : torch::nn::Module {
    explicit CrossBDQPolicyImpl(const CrossBDQPolicyConfig& cfg)
        : m_cfg(cfg)
        , lob_encoder(nullptr)
        , trade_encoder(nullptr)
        , fusion(nullptr)
        , value_head(nullptr)
        , adv_price(nullptr)
        , adv_ratio(nullptr) {
        auto F = (cfg.lob_feature_dim > 0 ? cfg.lob_feature_dim : 4 * cfg.lob_depth);
        (void)F;
        lob_encoder = register_module("lob_encoder", BDQ_LOBEncoder(cfg));
        trade_encoder = register_module("trade_encoder", BDQ_TradingFeatureEncoder(cfg.trade_hidden));
        fusion = register_module("fusion", BDQ_SharedStateFusion(cfg.lstm_hidden_size, cfg.trade_hidden, cfg.fused_dim, cfg.mlp_dropout));

        value_head = register_module("value_head", torch::nn::Sequential(
            torch::nn::Linear(cfg.fused_dim, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 1)
        ));
        adv_price = register_module("adv_price", torch::nn::Sequential(
            torch::nn::Linear(cfg.fused_dim, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, cfg.lob_depth + 1)
        ));
        adv_ratio = register_module("adv_ratio", torch::nn::Sequential(
            torch::nn::Linear(cfg.fused_dim, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, cfg.ratio_branches)
        ));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& lob_seq, const torch::Tensor& trade_vec) {
        auto lob_feat = lob_encoder->forward(lob_seq);
        auto trade_feat = trade_encoder->forward(trade_vec);
        auto shared = fusion->forward(lob_feat, trade_feat);
        auto V = value_head->forward(shared);                  // [B,1]
        auto A_p = adv_price->forward(shared);                 // [B,P]
        auto A_r = adv_ratio->forward(shared);                 // [B,R]
        auto A_p_centered = A_p - A_p.mean(/*dim=*/1, /*keepdim=*/true);
        auto A_r_centered = A_r - A_r.mean(/*dim=*/1, /*keepdim=*/true);
        auto Q_p = V + A_p_centered;
        auto Q_r = V + A_r_centered;
        return {Q_p, Q_r};
    }

    CrossBDQPolicyConfig m_cfg;
    BDQ_LOBEncoder lob_encoder{nullptr};
    BDQ_TradingFeatureEncoder trade_encoder{nullptr};
    BDQ_SharedStateFusion fusion{nullptr};
    torch::nn::Sequential value_head{nullptr};
    torch::nn::Sequential adv_price{nullptr};
    torch::nn::Sequential adv_ratio{nullptr};
};
TORCH_MODULE(CrossBDQPolicy);


