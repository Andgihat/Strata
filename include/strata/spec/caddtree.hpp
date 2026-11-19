#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::spec {

struct CaDDPlan {
    int T_prim = 1;       // primary trunk tokens (including root x0: 1 + drafts)
    int B = 0;            // alternative branch tokens
    int total_N = 1;      // T_prim + B
    float throughput = 0.0f;
    bool is_tree = false;
    int template_id = -1; // 0 for (4, 2), 1 for (5, 2)
};

/// Cost-Aware Diffusion Draft Trees planner (arXiv:2606.01813)
/// Optimizes token throughput theta = (1 + Phi(T)) / (C_d + C_v(|T|))
/// using a provably unimodal greedy search on a max-heap of candidate prefix probabilities.
inline CaDDPlan plan_caddtree(
    int num_prim_drafts,
    const int32_t* prim_tokens,
    const float* prim_probs,
    int num_alt_drafts,
    const int32_t* alt_tokens,
    int alt_div,
    int win_cap,
    double draft_conf = 0.0,
    float C_d = 3.7f,
    float alpha = 20.1f,
    float beta = 0.9f)
{
    CaDDPlan plan;
    if (win_cap < 2 || num_prim_drafts <= 0 || prim_tokens == nullptr || prim_probs == nullptr) {
        plan.T_prim = 1;
        plan.B = 0;
        plan.total_N = 1;
        plan.throughput = 1.0f / (C_d + alpha + beta);
        return plan;
    }

    const int max_N = std::min(win_cap, 8);
    auto C_v = [&](int N) -> float {
        return alpha + beta * (float) N;
    };

    // Cumulative prefix survival probabilities for primary chain
    std::vector<float> pi_prim(num_prim_drafts, 0.0f);
    float cum_p = 1.0f;
    for (int i = 0; i < num_prim_drafts; ++i) {
        cum_p *= std::clamp(prim_probs[i], 0.01f, 1.0f);
        pi_prim[i] = cum_p;
    }

    // Cumulative prefix survival probabilities for alternative branch
    std::vector<float> pi_alt(num_alt_drafts, 0.0f);
    if (alt_div == 0 && num_alt_drafts > 0 && alt_tokens != nullptr && alt_tokens[0] >= 0) {
        // When alt_div == 0, the alternative branch diverges at root token.
        // The probability of alt_tokens[0] is bounded by the residual probability at step 0.
        float p_alt0 = std::clamp(1.0f - prim_probs[0], 0.05f, 0.65f);
        pi_alt[0] = p_alt0;
        float cum_alt = p_alt0;
        for (int i = 1; i < num_alt_drafts; ++i) {
            if (alt_tokens[i] < 0) break;
            cum_alt *= std::clamp(prim_probs[i] * 0.9f, 0.05f, 0.95f);
            pi_alt[i] = cum_alt;
        }
    }

    struct CandidateNode {
        bool is_alt;
        int depth;       // index in prim or alt array (0-based)
        float pi;        // prefix probability
        bool operator<(const CandidateNode& o) const {
            return pi < o.pi; // max-heap
        }
    };

    std::priority_queue<CandidateNode> heap;
    if (num_prim_drafts > 0) {
        heap.push({false, 0, pi_prim[0]});
    }
    if (alt_div == 0 && num_alt_drafts > 0 && pi_alt[0] > 0.05f) {
        heap.push({true, 0, pi_alt[0]});
    }

    int cur_T_prim_drafts = 0;
    int cur_B = 0;
    int cur_N = 1; // root token always present
    float Phi = 0.0f;
    float theta = 1.0f / (C_d + C_v(1));

    while (!heap.empty() && cur_N < max_N) {
        CandidateNode node = heap.top();
        heap.pop();

        if (!node.is_alt && node.depth != cur_T_prim_drafts) continue;
        if (node.is_alt && node.depth != cur_B) continue;

        // Marginal throughput stopping rule: Delta Phi / Delta C > theta
        const float delta_C = beta;
        const float delta_Phi = node.pi;
        const float marginal_theta = delta_Phi / delta_C;

        if (marginal_theta <= theta) {
            // Unimodal stopping rule: marginal throughput is strictly declining
            break;
        }

        Phi += delta_Phi;
        cur_N += 1;
        theta = (1.0f + Phi) / (C_d + C_v(cur_N));

        if (!node.is_alt) {
            cur_T_prim_drafts = node.depth + 1;
            if (cur_T_prim_drafts < num_prim_drafts) {
                heap.push({false, cur_T_prim_drafts, pi_prim[cur_T_prim_drafts]});
            }
        } else {
            cur_B = node.depth + 1;
            if (cur_B < num_alt_drafts && cur_B < (int) pi_alt.size() && pi_alt[cur_B] > 0.0f) {
                heap.push({true, cur_B, pi_alt[cur_B]});
            }
        }
    }

    if (cur_T_prim_drafts < 1 && num_prim_drafts > 0) cur_T_prim_drafts = 1;

    // Map to canonical templates to guarantee bounded CUDA Graph count (<= 2 graphs):
    // Template 0: T_prim = 4 (1 root + 3 drafts), B = 2 (total 6)
    // Template 1: T_prim = 5 (1 root + 4 drafts), B = 2 (total 7)
    if (cur_B > 0 && max_N >= 6) {
        plan.is_tree = true;
        if (cur_T_prim_drafts >= 4 && max_N >= 7) {
            plan.T_prim = 5;
            plan.B = 2;
            plan.template_id = 1;
        } else {
            plan.T_prim = 4;
            plan.B = 2;
            plan.template_id = 0;
        }
        plan.total_N = plan.T_prim + plan.B;
    } else {
        plan.is_tree = false;
        plan.T_prim = cur_T_prim_drafts + 1; // include root token
        plan.B = 0;
        plan.total_N = plan.T_prim;
        plan.template_id = -1;
    }

    plan.throughput = theta;
    return plan;
}

} // namespace strata::spec
