// lib/nnue_reference.dart
//
// Float-reference NNUE (HalfKP), 2 accumulators, tempo-aware concatenation,
// incremental updates, and full training (MSE + Adam).
//
// Architecture: HalfKP[40960] -> M*2 -> K -> 1.
//
// Dependencies:
//   - chess3.dart         : Piece, PieceType, Color, Move
//   - constants.dart      : BITS_EP_CAPTURE, BITS_KSIDE_CASTLE, ...
//   - nnue_logic_batch2.dart : TrainingPosition(board, turn, target)

import 'dart:math';
import 'dart:typed_data';

import 'chess3.dart';
import 'constants.dart';
import 'nnue_logic_batch2.dart' show TrainingPosition;

class NNUERef {
  // Model dimensions
  static const int M = 256; // width of accumulator halves
  static const int K = 32; // width of hidden dense layer
  static const int NUM_FEATURES = 40960; // 64 * 64 * 5 * 2
  static const double SCALE_CP = 400.0; // network units <-> centipawns

  // Layers
  late final _Linear l0; // NUM_FEATURES -> M   (sparse: add columns)
  late final _Linear l1; // 2*M -> K            (dense)
  late final _Linear l2; // K   -> 1            (dense)

  // Adam optimizer state
  int _t = 0;

  // L1 moments (dense)
  late final Float64List _m1, _v1, _mb1, _vb1;

  // L2 moments (dense)
  late final Float64List _m2, _v2, _mb2, _vb2;

  // L0 bias moments
  late final Float64List _mb0, _vb0;

  // L0 sparse moment maps: featureIndex -> row vectors
  final Map<int, Float64List> _m0 = {};
  final Map<int, Float64List> _v0 = {};

  NNUERef({int seed = 42}) {
    final rng = Random(seed);
    l0 = _Linear(NUM_FEATURES, M)..heInit(rng);
    l1 = _Linear(2 * M, K)..heInit(rng);
    l2 = _Linear(K, 1)..heInit(rng);

    // Allocate Adam buffers
    _m1 = Float64List((2 * M) * K);
    _v1 = Float64List((2 * M) * K);
    _mb1 = Float64List(K);
    _vb1 = Float64List(K);

    _m2 = Float64List(K);
    _v2 = Float64List(K);
    _mb2 = Float64List(1);
    _vb2 = Float64List(1);

    _mb0 = Float64List(M);
    _vb0 = Float64List(M);
  }

  // ---------------------- Helpers ----------------------

  @pragma('vm:prefer-inline')
  static double crelu(double x) => x <= 0 ? 0 : (x >= 1 ? 1 : x);

  @pragma('vm:prefer-inline')
  static double _dCrelu(double x) => (x > 0 && x < 1) ? 1.0 : 0.0;

  @pragma('vm:prefer-inline')
  static int to64(int s) => ((s & 7) | ((s >> 4) << 3));

  @pragma('vm:prefer-inline')
  static int typeIndex(PieceType t) {
    switch (t) {
      case PieceType.PAWN:
        return 0;
      case PieceType.KNIGHT:
        return 1;
      case PieceType.BISHOP:
        return 2;
      case PieceType.ROOK:
        return 3;
      case PieceType.QUEEN:
        return 4;
      default:
        return 5; // king
    }
  }

  @pragma('vm:prefer-inline')
  static int halfKPIndex({
    required int kingSq, // 0..63
    required int pieceSq, // 0..63
    required Piece p,
    required Color perspective,
    required bool flip,
  }) {
    if (kingSq < 0 || kingSq > 63 || pieceSq < 0 || pieceSq > 63) return -1;

    final s = flip ? (pieceSq ^ 56) : pieceSq;
    final k = flip ? (kingSq ^ 56) : kingSq;

    final tIdx = typeIndex(p.type);
    if (tIdx >= 5) return -1; // exclude kings

    final usThem = (p.color == perspective) ? 0 : 1;
    final pIdx = tIdx * 2 + usThem;

    final idx = s + (pIdx + k * 10) * 64;
    return (idx >= 0 && idx < NUM_FEATURES) ? idx : -1;
  }

  static List<int> activeFeaturesForPerspective({
    required List<Piece?> board0x88,
    required Color perspective,
  }) {
    final king0x88 = board0x88.indexWhere(
      (q) => q?.type == PieceType.KING && q?.color == perspective,
    );
    if (king0x88 == -1) return const [];

    final k64 = to64(king0x88);
    final flip = (perspective == Color.BLACK);

    final out = <int>[];
    for (int sq = 0; sq < 128; sq++) {
      if ((sq & 0x88) != 0) continue;
      final p = board0x88[sq];
      if (p == null || p.type == PieceType.KING) continue;

      final i = halfKPIndex(
        kingSq: k64,
        pieceSq: to64(sq),
        p: p,
        perspective: perspective,
        flip: flip,
      );
      if (i != -1) out.add(i);
    }
    return out;
  }

  // ---------------- Accumulator ----------------

  NnueAccumulatorRef newAccumulator() => NnueAccumulatorRef(M);

  void refreshAccumulator(NnueAccumulatorRef acc, List<Piece?> board0x88) {
    final wf = activeFeaturesForPerspective(
      board0x88: board0x88,
      perspective: Color.WHITE,
    );
    final bf = activeFeaturesForPerspective(
      board0x88: board0x88,
      perspective: Color.BLACK,
    );

    acc.refresh(l0, wf, 0);
    acc.refresh(l0, bf, 1);
  }

  // -------------- Evaluation ------------------

  double evaluate(NnueAccumulatorRef acc, Color sideToMove) {
    final stm = (sideToMove == Color.WHITE) ? 0 : 1;
    final oth = 1 - stm;

    // [A_stm | A_oth]
    final inL1 = Float64List(2 * M);
    inL1.setRange(0, M, acc.v[stm]);
    inL1.setRange(M, 2 * M, acc.v[oth]);

    for (int i = 0; i < 2 * M; i++) inL1[i] = crelu(inL1[i]);

    final out1 = l1.forwardDense(inL1);
    for (int j = 0; j < K; j++) out1[j] = crelu(out1[j]);

    final out2 = l2.forwardDense(out1);
    return out2[0] * SCALE_CP;
  }

  // ---------------- Incremental Delta ----------------

  @pragma('vm:prefer-inline')
  static int _idxW({
    required int king64,
    required int sq64,
    required Piece p,
  }) => halfKPIndex(
    kingSq: king64,
    pieceSq: sq64,
    p: p,
    perspective: Color.WHITE,
    flip: false,
  );

  @pragma('vm:prefer-inline')
  static int _idxB({
    required int king64,
    required int sq64,
    required Piece p,
  }) => halfKPIndex(
    kingSq: king64,
    pieceSq: sq64,
    p: p,
    perspective: Color.BLACK,
    flip: true,
  );

  Map<String, List<int>> deltaForMove({
    required List<Piece?> boardBefore,
    required Move move,
  }) {
    final us = move.color;
    final them = (us == Color.WHITE) ? Color.BLACK : Color.WHITE;

    final wK0 = boardBefore.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    final bK0 = boardBefore.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );
    if (wK0 == -1 || bK0 == -1) {
      return {'addedW': [], 'removedW': [], 'addedB': [], 'removedB': []};
    }

    final wK = to64(wK0);
    final bK = to64(bK0);

    final from64 = to64(move.from);
    final to64sq = to64(move.to);

    final addedW = <int>[];
    final removedW = <int>[];
    final addedB = <int>[];
    final removedB = <int>[];

    void stage(int sq64, PieceType t, Color c, bool add) {
      final p = Piece(t, c);
      final wi = _idxW(king64: wK, sq64: sq64, p: p);
      final bi = _idxB(king64: bK, sq64: sq64, p: p);
      if (wi != -1) (add ? addedW : removedW).add(wi);
      if (bi != -1) (add ? addedB : removedB).add(bi);
    }

    stage(from64, move.piece, us, false);
    if (move.promotion != null) {
      stage(to64sq, move.promotion!, us, true);
    } else {
      stage(to64sq, move.piece, us, true);
    }

    if (move.captured != null) {
      int capSq64 = to64sq;
      if ((move.flags & BITS_EP_CAPTURE) != 0) {
        final cap0x = (us == Color.WHITE) ? (move.to + 16) : (move.to - 16);
        capSq64 = to64(cap0x);
      }
      stage(capSq64, move.captured!, them, false);
    }

    if ((move.flags & (BITS_KSIDE_CASTLE | BITS_QSIDE_CASTLE)) != 0) {
      late int rFrom;
      late int rTo;
      if ((move.flags & BITS_KSIDE_CASTLE) != 0) {
        rFrom = move.to + 1;
        rTo = move.to - 1;
      } else {
        rFrom = move.to - 2;
        rTo = move.to + 1;
      }
      stage(to64(rFrom), PieceType.ROOK, us, false);
      stage(to64(rTo), PieceType.ROOK, us, true);
    }

    return {
      'addedW': addedW,
      'removedW': removedW,
      'addedB': addedB,
      'removedB': removedB,
    };
  }

  // ---------------- Training (Adam + MSE) ----------------

  void trainBatch(List<TrainingPosition> batch, double lr) {
    if (batch.isEmpty) return;

    const double b1 = 0.9;
    const double b2 = 0.999;

    final gL2W = Float64List(K);
    double gL2B = 0;

    final gL1W = Float64List((2 * M) * K);
    final gL1B = Float64List(K);

    final Map<int, Float64List> gL0W = {};
    final gL0B = Float64List(M);

    for (final pos in batch) {
      // Recompute fresh accumulator for training
      final acc = newAccumulator();
      refreshAccumulator(acc, pos.board);

      final stm = (pos.turn == Color.WHITE) ? 0 : 1;
      final oth = 1 - stm;

      final inL1 = Float64List(2 * M);
      inL1.setRange(0, M, acc.v[stm]);
      inL1.setRange(M, 2 * M, acc.v[oth]);

      for (int i = 0; i < 2 * M; i++) inL1[i] = crelu(inL1[i]);

      final out1 = l1.forwardDense(inL1);
      for (int i = 0; i < K; i++) out1[i] = crelu(out1[i]);

      final pred = l2.forwardDense(out1)[0];
      final target = pos.target / SCALE_CP;

      final gradOut = 2.0 * (pred - target) / batch.length;

      // ----- Backprop L2 -----
      final gL2Out = Float64List(K);
      for (int j = 0; j < K; j++) {
        gL2Out[j] = gradOut * l2.weights[j][0] * _dCrelu(out1[j]);
        gL2W[j] += gradOut * out1[j];
      }
      gL2B += gradOut;

      // ----- Backprop L1 -----
      final gL1Out = Float64List(2 * M);

      for (int i = 0; i < 2 * M; i++) {
        double sum = 0;
        final base = i * K;
        for (int j = 0; j < K; j++) {
          sum += gL2Out[j] * l1.weights[i][j];
          gL1W[base + j] += gL2Out[j] * inL1[i];
        }
        gL1Out[i] = sum * _dCrelu(inL1[i]);
      }

      for (int j = 0; j < K; j++) gL1B[j] += gL2Out[j];

      // ----- L0 bias grads -----
      for (int j = 0; j < M; j++) {
        gL0B[j] += gL1Out[j] + gL1Out[M + j];
      }

      // ----- L0 sparse grads -----
      final wf = activeFeaturesForPerspective(
        board0x88: pos.board,
        perspective: Color.WHITE,
      );
      final bf = activeFeaturesForPerspective(
        board0x88: pos.board,
        perspective: Color.BLACK,
      );

      final wOffset = (stm == 0) ? 0 : M;
      final bOffset = (stm == 1) ? 0 : M;

      for (final f in wf) {
        final row = gL0W.putIfAbsent(f, () => Float64List(M));
        for (int j = 0; j < M; j++) row[j] += gL1Out[wOffset + j];
      }
      for (final f in bf) {
        final row = gL0W.putIfAbsent(f, () => Float64List(M));
        for (int j = 0; j < M; j++) row[j] += gL1Out[bOffset + j];
      }
    }

    // ---- Adam update ----

    _t++;
    final double bc1 = 1.0 - pow(b1, _t).toDouble();
    final double bc2 = 1.0 - pow(b2, _t).toDouble();

    // L2: K -> 1
    for (int i = 0; i < K; i++) {
      _m2[i] = b1 * _m2[i] + (1 - b1) * gL2W[i];
      _v2[i] = b2 * _v2[i] + (1 - b2) * (gL2W[i] * gL2W[i]);
      l2.weights[i][0] -= lr * (_m2[i] / bc1) / (sqrt(_v2[i] / bc2) + 1e-8);
    }
    _mb2[0] = b1 * _mb2[0] + (1 - b1) * gL2B;
    _vb2[0] = b2 * _vb2[0] + (1 - b2) * (gL2B * gL2B);
    l2.bias[0] -= lr * (_mb2[0] / bc1) / (sqrt(_vb2[0] / bc2) + 1e-8);

    // L1
    for (int i = 0; i < (2 * M) * K; i++) {
      _m1[i] = b1 * _m1[i] + (1 - b1) * gL1W[i];
      _v1[i] = b2 * _v1[i] + (1 - b2) * (gL1W[i] * gL1W[i]);
      final row = i ~/ K;
      final col = i % K;
      l1.weights[row][col] -= lr * (_m1[i] / bc1) / (sqrt(_v1[i] / bc2) + 1e-8);
    }
    for (int j = 0; j < K; j++) {
      _mb1[j] = b1 * _mb1[j] + (1 - b1) * gL1B[j];
      _vb1[j] = b2 * _vb1[j] + (1 - b2) * (gL1B[j] * gL1B[j]);
      l1.bias[j] -= lr * (_mb1[j] / bc1) / (sqrt(_vb1[j] / bc2) + 1e-8);
    }

    // L0 sparse
    gL0W.forEach((idx, grads) {
      final mRow = _m0.putIfAbsent(idx, () => Float64List(M));
      final vRow = _v0.putIfAbsent(idx, () => Float64List(M));
      final wRow = l0.weights[idx];

      for (int j = 0; j < M; j++) {
        mRow[j] = b1 * mRow[j] + (1 - b1) * grads[j];
        vRow[j] = b2 * vRow[j] + (1 - b2) * (grads[j] * grads[j]);
        wRow[j] -= lr * (mRow[j] / bc1) / (sqrt(vRow[j] / bc2) + 1e-8);
      }
    });

    // L0 bias
    for (int j = 0; j < M; j++) {
      _mb0[j] = b1 * _mb0[j] + (1 - b1) * gL0B[j];
      _vb0[j] = b2 * _vb0[j] + (1 - b2) * (gL0B[j] * gL0B[j]);
      l0.bias[j] -= lr * (_mb0[j] / bc1) / (sqrt(_vb0[j] / bc2) + 1e-8);
    }
  }
}

// =========================================================
// Dense linear layer
// =========================================================
class _Linear {
  final int inSize, outSize;
  final List<Float64List> weights; // column-major: weights[c][r]
  final Float64List bias;

  _Linear(this.inSize, this.outSize)
    : weights = List.generate(inSize, (_) => Float64List(outSize)),
      bias = Float64List(outSize);

  void heInit(Random rng) {
    final s = sqrt(2.0 / inSize) * 1.1;
    for (int c = 0; c < inSize; c++) {
      final col = weights[c];
      for (int r = 0; r < outSize; r++) {
        col[r] = (rng.nextDouble() * 2 - 1) * s;
      }
    }
    for (int r = 0; r < outSize; r++) bias[r] = 0;
  }

  Float64List forwardDense(Float64List input) {
    final out = Float64List.fromList(bias);
    for (int c = 0; c < inSize; c++) {
      final x = input[c];
      if (x == 0) continue;
      final col = weights[c];
      for (int r = 0; r < outSize; r++) out[r] += x * col[r];
    }
    return out;
  }
}

// =========================================================
// Accumulator
// =========================================================
class NnueAccumulatorRef {
  final List<Float64List> v; // v[0] = white, v[1] = black
  final int m;

  NnueAccumulatorRef(this.m) : v = [Float64List(m), Float64List(m)];

  void refresh(_Linear l0, List<int> active, int p) {
    v[p].setAll(0, l0.bias);
    for (final idx in active) {
      final col = l0.weights[idx];
      for (int i = 0; i < m; i++) {
        v[p][i] += col[i];
      }
    }
  }

  void update(_Linear l0, List<int> added, List<int> removed, int p) {
    for (final idx in removed) {
      final col = l0.weights[idx];
      for (int i = 0; i < m; i++) v[p][i] -= col[i];
    }
    for (final idx in added) {
      final col = l0.weights[idx];
      for (int i = 0; i < m; i++) v[p][i] += col[i];
    }
  }
}
