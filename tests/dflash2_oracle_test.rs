//! S2F unit tests — the DFlash2 reference oracle, on a REDUCED-shape config so CI-style reruns
//! are cheap (the exact same code paths as the full-size `--probe-dflash2`). Covers the workdoc
//! §3 discipline: determinism, wiring (anti-empty-compare), incremental==batch taps + KV,
//! SWA-window boundary, conv causality, selector structure, and piecewise availability. The
//! full-shape end-to-end lives in the probe (real artifact + golden dump).

use gb10_inference::dflash2::oracle::{
    Dflash2Config, Dflash2Oracle, Dflash2Weights, LayerWeights, ConvWeights, RoundCtx,
};
use gb10_inference::dflash2::synth::{SynthRng, SyntheticTables};

/// A reduced but structurally faithful config: GQA 4Q/2KV × hd32, hidden 128 (= 4×32), 2 layers,
/// vocab 4096, selector rank 16, conv groups 16 (group size 8), window 16 — block 8 / MASK id /
/// θ=1e7 / non-causal identical to the real anatomy.
fn reduced_cfg() -> Dflash2Config {
    Dflash2Config {
        hidden: 128,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        inter: 256,
        vocab: 4096,
        n_layers: 2,
        block: 8,
        mask_token_id: 248_070,
        rope_theta: 1e7,
        rms_eps: 1e-6,
        sliding_window: 16,
        is_causal: false,
        conv_kernel: 2,
        conv_group: 8,
        selector_rank: 16,
        selector_top_k: 16,
        n_taps: 5,
    }
}

fn reduced_weights(cfg: &Dflash2Config) -> Dflash2Weights {
    let mut rng = SynthRng::new(0xDF2A_0000_1111_2222);
    let norm = |rng: &mut SynthRng, n: usize| -> Vec<f32> {
        (0..n).map(|_| 0.5 + 0.5 * rng.uniform()).collect()
    };
    let lin = |rng: &mut SynthRng, outn: usize, inn: usize| -> Vec<f32> {
        let scale = 1.0 / (inn as f32).sqrt();
        (0..outn * inn).map(|_| rng.normal() * scale).collect()
    };
    let (h, nh, nkv, hd, inter, rank, vocab, k, groups) = (
        cfg.hidden, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.inter,
        cfg.selector_rank, cfg.vocab, cfg.conv_kernel, cfg.hidden / cfg.conv_group,
    );
    let conv = |rng: &mut SynthRng| -> ConvWeights {
        ConvWeights {
            base_kernel: lin(rng, 2 * k, h),
            kernel_projection: lin(rng, 2 * k * groups, h),
        }
    };
    let mut layers = Vec::new();
    for _ in 0..cfg.n_layers {
        layers.push(LayerWeights {
            q_proj: lin(&mut rng, nh * hd, h),
            k_proj: lin(&mut rng, nkv * hd, h),
            v_proj: lin(&mut rng, nkv * hd, h),
            o_proj: lin(&mut rng, h, nh * hd),
            q_norm: norm(&mut rng, hd),
            k_norm: norm(&mut rng, hd),
            input_ln: norm(&mut rng, h),
            post_ln: norm(&mut rng, h),
            gate_proj: lin(&mut rng, inter, h),
            up_proj: lin(&mut rng, inter, h),
            down_proj: lin(&mut rng, h, inter),
            attention_conv: conv(&mut rng),
            mlp_conv: conv(&mut rng),
        });
    }
    Dflash2Weights {
        layers,
        fc: lin(&mut rng, h, 5 * h),
        hidden_norm: norm(&mut rng, h),
        norm: norm(&mut rng, h),
        hidden_projection: lin(&mut rng, rank, h),
        predecessor_codebook: lin(&mut rng, vocab, rank),
        successor_codebook: lin(&mut rng, vocab, rank),
    }
}

fn reduced_ctx(cfg: &Dflash2Config, c: usize) -> RoundCtx {
    let taps_gen = SyntheticTables::new(gb10_inference::dflash2::SYNTH_TAP_SEED);
    let tap_dim = cfg.n_taps * cfg.hidden;
    let scale = 1.0 / (tap_dim as f32).sqrt();
    let mut taps = Vec::with_capacity(c * tap_dim);
    for i in 0..c {
        taps.extend_from_slice(&taps_gen.row(SyntheticTables::TABLE_TAPS, i as u32, tap_dim, scale));
    }
    RoundCtx { tap_hiddens: taps, anchor: 57 }
}

#[test]
fn inventory_reconciles_to_the_real_counts() {
    let inv = gb10_inference::dflash2::inventory();
    assert_eq!(inv.len(), 81);
    let mut names: Vec<&String> = inv.iter().map(|(n, _)| n).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 81, "inventory names must be unique");
    assert_eq!(gb10_inference::dflash2::reconcile_params(), 1_924_404_480);
    assert_eq!(Dflash2Config::default().n_params(), 1_924_404_480);
}

#[test]
fn determinism_two_rounds_bit_identical() {
    let cfg = reduced_cfg();
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), reduced_weights(&cfg)).unwrap();
    let ctx = reduced_ctx(&cfg, 11);
    let a = oracle.run_round(&ctx);
    let b = oracle.run_round(&ctx);
    assert!(a.th == b.th && a.layer_hiddens == b.layer_hiddens && a.h == b.h
        && a.logits == b.logits && a.select.tokens == b.select.tokens
        && a.select.candidates == b.select.candidates && a.select.scores == b.select.scores);
}

#[test]
fn wiring_sign_flip_changes_outputs() {
    let cfg = reduced_cfg();
    let mut w = reduced_weights(&cfg);
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), w.clone()).unwrap();
    let ctx = reduced_ctx(&cfg, 11);
    let a = oracle.run_round(&ctx);
    let idx = w.layers[0].q_proj.iter().position(|&x| x != 0.0).unwrap();
    w.layers[0].q_proj[idx] = -w.layers[0].q_proj[idx];
    let oracle_p = Dflash2Oracle::from_weights(cfg.clone(), w).unwrap();
    let b = oracle_p.run_round(&ctx);
    assert!(a.logits != b.logits || a.h != b.h || a.select.tokens != b.select.tokens);
}

#[test]
fn incremental_equals_batch_taps_and_kv() {
    let cfg = reduced_cfg();
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), reduced_weights(&cfg)).unwrap();
    let ctx = reduced_ctx(&cfg, 11);
    let tap_dim = cfg.n_taps * cfg.hidden;
    let th_batch = oracle.tap_project(&ctx.tap_hiddens, 11);
    let mut th_inc = Vec::new();
    for i in 0..11 {
        th_inc.extend_from_slice(&oracle.tap_project(
            &ctx.tap_hiddens[i * tap_dim..(i + 1) * tap_dim], 1));
    }
    assert_eq!(th_batch, th_inc, "tap projection must chunk bit-identically");
    let kv_batch = oracle.draft_kv_write(&th_batch, 11, 0);
    let mut kv_inc = oracle.draft_kv_write(&th_batch[..4 * cfg.hidden], 4, 0);
    kv_inc.append(oracle.draft_kv_write(&th_batch[4 * cfg.hidden..], 7, 4));
    for (x, y) in kv_batch.layers.iter().zip(kv_inc.layers.iter()) {
        assert!(x.k == y.k && x.v == y.v && x.m == y.m, "ctx KV must chunk bit-identically");
    }
}

#[test]
fn sliding_window_boundary_is_exact() {
    // window 16, C = 40: block query i (key pos 40+i) sees ctx rows j ≥ 40+i−16+1 = 25+i.
    // Row 24 is masked for ALL queries; row 25 is visible to query 0.
    let cfg = reduced_cfg();
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), reduced_weights(&cfg)).unwrap();
    let ctx = reduced_ctx(&cfg, 40);
    let base = oracle.run_round(&ctx);
    let tap_dim = cfg.n_taps * cfg.hidden;
    let mut masked = ctx.tap_hiddens.clone();
    masked[24 * tap_dim] = 1.2345678;
    let out_masked = oracle.run_round(&RoundCtx { tap_hiddens: masked, anchor: ctx.anchor });
    assert!(out_masked.h == base.h && out_masked.logits == base.logits
        && out_masked.select.tokens == base.select.tokens,
        "a masked ctx row must not affect the block (bit-identical)");
    let mut visible = ctx.tap_hiddens.clone();
    visible[25 * tap_dim] = 1.2345678;
    let out_visible = oracle.run_round(&RoundCtx { tap_hiddens: visible, anchor: ctx.anchor });
    assert!(out_visible.h != base.h || out_visible.logits != base.logits,
        "a visible ctx row must affect the block");
}

#[test]
fn conv_is_causal_and_block_local() {
    let cfg = reduced_cfg();
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), reduced_weights(&cfg)).unwrap();
    let conv = &oracle.weights.layers[0].attention_conv;
    let n = cfg.block;
    let mut rng = SynthRng::new(0xABCD);
    let x: Vec<f32> = (0..n * cfg.hidden).map(|_| rng.normal()).collect();
    let prep = oracle.conv_prepare(conv, &x, n);
    // row 0 depends only on x[0] (the tap-1 shift sees the zero pad).
    let mut x2 = x.clone();
    for v in &mut x2[..cfg.hidden] { *v += 1.0; }
    let prep2 = oracle.conv_prepare(conv, &x2, n);
    assert_ne!(prep.x_conv[..cfg.hidden], prep2.x_conv[..cfg.hidden],
        "row 0 must respond to its own row");
    // changing row 5 must not touch rows 0..5 (causality), must touch rows 5..7.
    let mut x3 = x.clone();
    for v in &mut x3[5 * cfg.hidden..6 * cfg.hidden] { *v += 1.0; }
    let prep3 = oracle.conv_prepare(conv, &x3, n);
    assert_eq!(prep.x_conv[..5 * cfg.hidden], prep3.x_conv[..5 * cfg.hidden],
        "conv must be causal (later rows never leak upward)");
    assert_ne!(prep.x_conv[5 * cfg.hidden..], prep3.x_conv[5 * cfg.hidden..],
        "conv must respond to the changed row");
    // the dynamic taps actually modulate: prepare(x) != plain base-only convolve.
    let k = cfg.conv_kernel;
    let zeros = vec![0.0f32; n * k * (cfg.hidden / cfg.conv_group)];
    let base0 = &conv.base_kernel[..k * cfg.hidden];
    let static_only = oracle.convolve(&x, &zeros, base0, n);
    assert_ne!(prep.x_conv, static_only, "the dynamic kernel taps must be wired");
    // finish transforms its input too.
    let fin = oracle.conv_finish(conv, &prep.x_conv, &prep.dyn_hold, n);
    assert_ne!(fin, prep.x_conv);
    assert!(fin.iter().all(|v| v.is_finite()));
}

#[test]
fn rope_position_zero_is_identity() {
    let cfg = reduced_cfg();
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), reduced_weights(&cfg)).unwrap();
    // a layer's q at position 0: roping is the identity (cos 1 / sin 0) — verify via attn on a
    // zero-ctx round is overkill; check the table property through a tiny direct probe instead:
    // run a 1-row kv write at pos_start 0 and at pos_start 1; the k rows must differ (RoPE wired)
    // and the pos-0 row must equal the un-roped k_proj+k_norm output.
    let th: Vec<f32> = {
        let mut rng = SynthRng::new(0x9999);
        (0..cfg.hidden).map(|_| rng.normal()).collect()
    };
    let kv0 = oracle.draft_kv_write(&th, 1, 0);
    let kv1 = oracle.draft_kv_write(&th, 1, 1);
    assert_ne!(kv0.layers[0].k, kv1.layers[0].k, "RoPE must rotate k at nonzero positions");
    // reference un-roped k: k_proj + k_norm, computed by hand.
    let l0 = &oracle.weights.layers[0];
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let mut k = vec![0.0f32; nkv * hd];
    for o in 0..nkv * hd {
        let mut acc = 0.0f32;
        for i in 0..cfg.hidden {
            acc += l0.k_proj[o * cfg.hidden + i] * th[i];
        }
        k[o] = acc;
    }
    for h in 0..nkv {
        let base = h * hd;
        let mut ss = 0.0f32;
        for d in 0..hd { ss += k[base + d] * k[base + d]; }
        let inv = 1.0 / (ss / hd as f32 + cfg.rms_eps).sqrt();
        for d in 0..hd { k[base + d] *= inv * l0.k_norm[d]; }
    }
    assert_eq!(k, kv0.layers[0].k, "position 0 RoPE must be the identity");
}

#[test]
fn selector_structure_and_chain() {
    let cfg = reduced_cfg();
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), reduced_weights(&cfg)).unwrap();
    let ctx = reduced_ctx(&cfg, 11);
    let out = oracle.run_round(&ctx);
    assert!(out.logits.iter().all(|x| x.is_finite()));
    assert!(out.h.iter().all(|x| x.is_finite()));
    assert!(out.select.tokens.iter().all(|&t| (t as usize) < cfg.vocab));
    // candidates == independent full-sort top-16 (value desc, id asc); unary == row values.
    for p in 0..7 {
        let row = &out.logits[p * cfg.vocab..(p + 1) * cfg.vocab];
        let mut order: Vec<(f32, u32)> =
            row.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
        order.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        for kk in 0..16 {
            assert_eq!(order[kk].1, out.select.candidates[p][kk], "candidate order mismatch");
            assert_eq!(order[kk].0, out.select.unary[p][kk], "unary mismatch");
        }
    }
    // chain wiring: perturbing the anchor's predecessor-codebook row changes the scores.
    let mut w2 = reduced_weights(&cfg);
    let anchor = ctx.anchor as usize;
    let idx = w2.predecessor_codebook[anchor * cfg.selector_rank..]
        .iter().position(|&x| x != 0.0).unwrap();
    w2.predecessor_codebook[anchor * cfg.selector_rank + idx] *= -1.0;
    let oracle2 = Dflash2Oracle::from_weights(cfg.clone(), w2).unwrap();
    let out2 = oracle2.run_round(&ctx);
    assert!(out2.select.scores != out.select.scores,
        "the predecessor codebook must feed the chain scores");
}

#[test]
fn piecewise_composition_equals_run_round() {
    let cfg = reduced_cfg();
    let oracle = Dflash2Oracle::from_weights(cfg.clone(), reduced_weights(&cfg)).unwrap();
    let ctx = reduced_ctx(&cfg, 11);
    let c = 11usize;
    let full = oracle.run_round(&ctx);
    let th = oracle.tap_project(&ctx.tap_hiddens, c);
    let kv = oracle.draft_kv_write(&th, c, 0);
    let hidden = cfg.hidden;
    let scale = 1.0 / (hidden as f32).sqrt();
    let synth = SyntheticTables::new(gb10_inference::dflash2::SYNTH_EMBED_HEAD_SEED);
    let mut emb = Vec::with_capacity(cfg.block * hidden);
    emb.extend_from_slice(&synth.row(SyntheticTables::TABLE_EMBED, ctx.anchor, hidden, scale));
    for _ in 1..cfg.block {
        emb.extend_from_slice(&synth.row(SyntheticTables::TABLE_EMBED, cfg.mask_token_id, hidden, scale));
    }
    let block_pos: Vec<usize> = (c..c + cfg.block).collect();
    let h1 = oracle.layer_forward(0, &emb, &kv.layers[0], &block_pos);
    assert_eq!(h1, full.layer_hiddens[0], "layer_forward(0) != run_round layer 0");
    let (layers, h) = oracle.backbone_forward(&emb, &kv, c);
    assert_eq!(layers, full.layer_hiddens);
    assert_eq!(h, full.h);
    let h_sel = &h[hidden..cfg.block * hidden];
    let logits = oracle.logits(h_sel, 7);
    assert_eq!(logits, full.logits);
    let sel = oracle.select_path(h_sel, &logits, ctx.anchor);
    assert_eq!(sel.tokens, full.select.tokens);
    assert_eq!(sel.scores, full.select.scores);
    // select_chain with select_path's own candidates/unary reproduces the path.
    let sel2 = oracle.select_chain(h_sel, &sel.candidates, &sel.unary, ctx.anchor);
    assert_eq!(sel2.tokens, sel.tokens);
}
