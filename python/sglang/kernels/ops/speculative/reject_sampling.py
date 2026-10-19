import triton
import triton.language as tl


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Relaxed-acceptance knobs (see the accept condition below). Runtime
    # scalars, not constexpr, so changing them does not retrigger a JIT
    # compile of this kernel.
    threshold_single,
    threshold_acc,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        offset_draft = (
            (pid * stride_dp_b)
            + (cur_prob_row * stride_dp_s)
            + (draft_token * stride_dp_v)
        )

        p = tl.load(TargetProbs + offset_prob)
        q = tl.load(DraftProbs + offset_draft)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        # X was sampled from q, so q(X) has to be a positive probability.
        # Anything else means this row is not the distribution X came from, and
        # `coin * q < p` would then accept unconditionally -- -inf < p for an
        # -inf q, 0 < p for a zero one, and the range guard the residual passes
        # use lets zero through. Reject instead: the residual path resamples
        # from the target, which is the safe direction to fail in.
        q_is_prob = (q > 0.0) & (q <= 1.0)

        # Relaxed acceptance (lossy above the defaults).
        #
        #   threshold_acc scales the rejection-sampling bar: the exact rule
        #   accepts with probability min(1, p/q), i.e. coin*q < p. Dividing q
        #   by threshold_acc raises that to min(1, p/(q*threshold_acc)), so
        #   threshold_acc < 1 accepts strictly more often.
        #
        #   threshold_single is an unconditional accept for tokens the target
        #   already likes: any draft token with target probability >= it is
        #   taken regardless of the coin.
        #
        # At the 1.0/1.0 defaults this is byte-for-byte the previous rule:
        # coin*q*1.0 < p, and p >= 1.0 only when the target is fully
        # saturated on that token (in which case coin*q < p accepts anyway).
        accept_relaxed = coin * q * threshold_acc < p
        accept_single = p >= threshold_single
        if q_is_prob & (accept_relaxed | accept_single):
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    # Pass 1: sum and track a deterministic fallback. Pass 1 and the CDF pass
    # use different reduction trees, so a final uniform sample near 1 can
    # expose a rounding gap where the CDF never becomes strictly greater than
    # target_u. Returning the last vocabulary id in that case is arbitrary and
    # can select a zero-probability token; the largest residual is a safe,
    # deterministic fallback.
    best_idx = VOCAB_SIZE - 1
    best_val = -1.0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
            q_val = tl.load(q_ptr, mask=mask, other=0.0)
            # Treat any non-probability q (NaN, +-inf, negative) as 0: the
            # residual falls back to p. A comparison against NaN is false, so
            # the range test rejects it along with the infinities.
            q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)

        norm_sum += tl.sum(val)

        block_max = tl.max(val, axis=0)
        if block_max > best_val:
            best_val = block_max
            best_idx = v_start + tl.argmax(val, axis=0)

    # Pass 2: CDF. Use pass 1's argmax if independent floating-point reductions
    # leave the CDF just below target_u, or if the residual is degenerate.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = best_idx
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Same guard as pass 1.
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        float(threshold_single),
        # Guard against a zero/negative divisor turning the accept test into
        # "accept everything" by accident; callers already clamp, this is
        # defence in depth for direct kernel users.
        max(float(threshold_acc), 1e-9),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
    )
