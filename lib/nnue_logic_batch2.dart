import 'dart:math';
import 'dart:typed_data';
import 'chess3.dart';

class TrainingPosition {
  final List<Piece?> board; // 0x88 board (length 128)
  final Color turn;
  final double target;
  TrainingPosition(this.board, this.turn, this.target);
}

class LinearLayer {
  final int numInputs, numOutputs;
  final List<Float64List> weights;
  final Float64List bias;

  LinearLayer(this.numInputs, this.numOutputs)
    : weights = List.generate(numInputs, (_) => Float64List(numOutputs)),
      bias = Float64List(numOutputs);

  void randomize(Random rng) {
    // He-like init
    final double scale = sqrt(2.0 / numInputs) * 1.1;
    for (var i = 0; i < numInputs; i++) {
      final row = weights[i];
      for (var j = 0; j < numOutputs; j++) {
        row[j] = (rng.nextDouble() * 2 - 1) * scale;
      }
    }
    // Neutral bias (you can set to 0.02 like before if you prefer)
    for (var i = 0; i < numOutputs; i++) bias[i] = 0.0;
  }
}

class NnueAccumulator {
  // Two halves: [whitePerspective, blackPerspective]
  final List<Float64List> v;
  final int m;
  NnueAccumulator(this.m) : v = [Float64List(m), Float64List(m)];

  void refresh(LinearLayer l0, List<int> active, int p) {
    // Seed with l0 bias, then add each active feature row
    v[p].setAll(0, l0.bias);
    for (final idx in active) {
      if (idx >= 0 && idx < l0.numInputs) {
        final w = l0.weights[idx];
        for (var i = 0; i < m; i++) v[p][i] += w[i];
      }
    }
  }

  void update(LinearLayer l0, List<int> added, List<int> removed, int p) {
    for (final idx in removed) {
      if (idx >= 0 && idx < l0.numInputs) {
        final w = l0.weights[idx];
        for (var i = 0; i < m; i++) v[p][i] -= w[i];
      }
    }
    for (final idx in added) {
      if (idx >= 0 && idx < l0.numInputs) {
        final w = l0.weights[idx];
        for (var i = 0; i < m; i++) v[p][i] += w[i];
      }
    }
  }

  NnueAccumulator copy() {
    final newAcc = NnueAccumulator(m);
    newAcc.v[0].setAll(0, v[0]);
    newAcc.v[1].setAll(0, v[1]);
    return newAcc;
  }
}

class NNUE {
  static const int M = 256; // neurons per accumulator half
  static const int K = 32; // hidden layer width
  static const int NUM_FEATURES = 40960; // 64 * 640 = 40960 (HalfKP)
  static const double SCALE = 400.0; // convert net units <-> centipawns

  late LinearLayer l0, l1, l2;

  // Adam moments for l0 weights (sparse rows), l0.bias, l1 (flattened), l2, and biases.
  late List<Float64List> m0, v0; // [NUM_FEATURES][M]
  late Float64List mb0, vb0; // l0.bias (M)
  late Float64List m1, v1, mb1, vb1; // l1 weights ((2M*K)) and bias (K)
  late Float64List m2,
      v2,
      mb2,
      vb2; // l2 weights (K->1 flattened K) and bias (1)

  int t = 0; // Adam step

  NNUE() {
    final rng = Random(42);
    l0 = LinearLayer(NUM_FEATURES, M)..randomize(rng);
    l1 = LinearLayer(2 * M, K)..randomize(rng);
    l2 = LinearLayer(K, 1)..randomize(rng);
    _initAdam();
  }

  void _initAdam() {
    m0 = List.generate(NUM_FEATURES, (_) => Float64List(M));
    v0 = List.generate(NUM_FEATURES, (_) => Float64List(M));
    mb0 = Float64List(M);
    vb0 = Float64List(M);

    m1 = Float64List((2 * M) * K);
    v1 = Float64List((2 * M) * K);
    mb1 = Float64List(K);
    vb1 = Float64List(K);

    m2 = Float64List(K);
    v2 = Float64List(K);
    mb2 = Float64List(1);
    vb2 = Float64List(1);
  }

  @pragma('vm:prefer-inline')
  double _crelu(double x) => x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);

  @pragma('vm:prefer-inline')
  double _creluDeriv(double x) => (x > 0.0 && x < 1.0) ? 1.0 : 0.0;

  int getTypeIndex(PieceType type) {
    if (type == PieceType.PAWN) return 0;
    if (type == PieceType.KNIGHT) return 1;
    if (type == PieceType.BISHOP) return 2;
    if (type == PieceType.ROOK) return 3;
    if (type == PieceType.QUEEN) return 4;
    return 5; // King
  }

  /// Convert 0x88 square (0..127 valid squares on 0x88) to 0..63
  @pragma('vm:prefer-inline')
  static int _to64(int sq0x88) => ((sq0x88 & 7) | ((sq0x88 >> 4) << 3));

  void trainBatch(List<TrainingPosition> batch, double lr) {
    if (batch.isEmpty) return;
    t++;
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    final double bc1 = 1.0 - (pow(b1, t) as double);
    final double bc2 = 1.0 - (pow(b2, t) as double);

    final gL2W = Float64List(K);
    var gL2B = 0.0;
    final gL1W = Float64List((2 * M) * K);
    final gL1B = Float64List(K);
    final gL0W = <int, Float64List>{}; // sparse: featureIndex -> grad[M]
    final gL0B = Float64List(M); // grad for l0.bias

    for (final pos in batch) {
      final acc = NnueAccumulator(M);
      _refreshAccumulator(acc, pos.board);

      final stmIdx = pos.turn == Color.WHITE ? 0 : 1;

      // Input to hidden (2*M): [stm half | opp half]
      final inL1 = Float64List(2 * M);
      inL1.setRange(0, M, acc.v[stmIdx]);
      inL1.setRange(M, 2 * M, acc.v[1 - stmIdx]);

      // Act1 = crelu(inL1)
      final act1 = Float64List(2 * M);
      for (int i = 0; i < 2 * M; i++) act1[i] = _crelu(inL1[i]);

      // Hidden pre-activation and activation
      final out1 = _forwardDense(l1, act1);
      final act2 = Float64List(K);
      for (int i = 0; i < K; i++) act2[i] = _crelu(out1[i]);

      // Output
      final pred = _forwardDense(l2, act2)[0];
      final target = pos.target / SCALE; // network units
      final gradOut = 2.0 * (pred - target) / batch.length;

      // Backprop Layer 2 (K -> 1)
      final gL2Out = Float64List(K); // dL/d(out1[i]) via chain
      for (int i = 0; i < K; i++) {
        gL2Out[i] = gradOut * l2.weights[i][0] * _creluDeriv(out1[i]);
        gL2W[i] += gradOut * act2[i];
      }
      gL2B += gradOut;

      // Backprop Layer 1 (2M -> K)
      final gL1Out = Float64List(2 * M); // dL/d(inL1[i])
      for (int i = 0; i < 2 * M; i++) {
        double sum = 0.0;
        for (int j = 0; j < K; j++) {
          sum += gL2Out[j] * l1.weights[i][j];
          gL1W[i * K + j] += gL2Out[j] * act1[i];
        }
        gL1Out[i] = sum * _creluDeriv(inL1[i]);
      }
      for (int j = 0; j < K; j++) gL1B[j] += gL2Out[j];

      // Accumulate l0.bias gradient: both halves contribute
      for (int j = 0; j < M; ++j) {
        gL0B[j] += gL1Out[j] + gL1Out[j + M];
      }

      // Backprop to L0 (sparse rows via feature indices)
      _accumulateL0(pos, gL1Out, gL0W);
    }

    _applyAdamUpdates(
      lr,
      b1,
      b2,
      bc1,
      bc2,
      eps,
      gL2W,
      gL2B,
      gL1W,
      gL1B,
      gL0W,
      gL0B,
    );
  }

  void _accumulateL0(
    TrainingPosition pos,
    Float64List gL1Out,
    Map<int, Float64List> gL0W,
  ) {
    // Find kings in 0x88 space then convert to 0..63 for feature index
    final wK0x88 = pos.board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    final bK0x88 = pos.board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );
    if (wK0x88 == -1 || bK0x88 == -1) return;

    final wK = _to64(wK0x88);
    final bK = _to64(bK0x88);

    for (int sq = 0; sq < 128; ++sq) {
      if ((sq & 0x88) != 0) continue; // skip 0x88 padding
      final p = pos.board[sq];
      if (p == null || p.type == PieceType.KING) continue; // skip kings

      final s64 = _to64(sq);
      final wf = getHalfKPIndex(wK, s64, p, false);
      final bf = getHalfKPIndex(bK, s64, p, true);

      if (wf != -1) {
        final v = gL0W.putIfAbsent(wf, () => Float64List(M));
        for (int j = 0; j < M; j++) v[j] += gL1Out[j];
      }
      if (bf != -1) {
        final v = gL0W.putIfAbsent(bf, () => Float64List(M));
        for (int j = 0; j < M; j++) v[j] += gL1Out[j + M];
      }
    }
  }

  void _applyAdamUpdates(
    double lr,
    double b1,
    double b2,
    double bc1,
    double bc2,
    double eps,
    Float64List gL2W,
    double gL2B,
    Float64List gL1W,
    Float64List gL1B,
    Map<int, Float64List> gL0W,
    Float64List gL0B,
  ) {
    // --- Layer 2 (K -> 1) ---
    for (int i = 0; i < K; i++) {
      m2[i] = b1 * m2[i] + (1.0 - b1) * gL2W[i];
      v2[i] = b2 * v2[i] + (1.0 - b2) * (gL2W[i] * gL2W[i]);
      l2.weights[i][0] -= lr * (m2[i] / bc1) / (sqrt(v2[i] / bc2) + eps);
    }
    mb2[0] = b1 * mb2[0] + (1.0 - b1) * gL2B;
    vb2[0] = b2 * vb2[0] + (1.0 - b2) * (gL2B * gL2B);
    l2.bias[0] -= lr * (mb2[0] / bc1) / (sqrt(vb2[0] / bc2) + eps);

    // --- Layer 1 (2M -> K) ---
    for (int i = 0; i < (2 * M) * K; i++) {
      m1[i] = b1 * m1[i] + (1.0 - b1) * gL1W[i];
      v1[i] = b2 * v1[i] + (1.0 - b2) * (gL1W[i] * gL1W[i]);
      final row = i ~/ K;
      final col = i % K;
      l1.weights[row][col] -= lr * (m1[i] / bc1) / (sqrt(v1[i] / bc2) + eps);
    }
    for (int j = 0; j < K; j++) {
      mb1[j] = b1 * mb1[j] + (1.0 - b1) * gL1B[j];
      vb1[j] = b2 * vb1[j] + (1.0 - b2) * (gL1B[j] * gL1B[j]);
      l1.bias[j] -= lr * (mb1[j] / bc1) / (sqrt(vb1[j] / bc2) + eps);
    }

    // --- Layer 0 (features -> M) ---
    // weights: sparse rows only
    gL0W.forEach((idx, grads) {
      if (idx >= 0 && idx < NUM_FEATURES) {
        final m0row = m0[idx];
        final v0row = v0[idx];
        final wrow = l0.weights[idx];
        for (int j = 0; j < M; j++) {
          m0row[j] = b1 * m0row[j] + (1.0 - b1) * grads[j];
          v0row[j] = b2 * v0row[j] + (1.0 - b2) * (grads[j] * grads[j]);
          wrow[j] -= lr * (m0row[j] / bc1) / (sqrt(v0row[j] / bc2) + eps);
        }
      }
    });

    // bias: full M vector
    for (int j = 0; j < M; ++j) {
      mb0[j] = b1 * mb0[j] + (1.0 - b1) * gL0B[j];
      vb0[j] = b2 * vb0[j] + (1.0 - b2) * (gL0B[j] * gL0B[j]);
      l0.bias[j] -= lr * (mb0[j] / bc1) / (sqrt(vb0[j] / bc2) + eps);
    }
  }

  /// HalfKP feature index. Inputs must be 0..63 squares.
  @pragma('vm:prefer-inline')
  int getHalfKPIndex(int kingSq, int pieceSq, Piece piece, bool flip) {
    if (kingSq < 0 || kingSq > 63 || pieceSq < 0 || pieceSq > 63) return -1;

    final s = flip ? (pieceSq ^ 56) : pieceSq;
    final k = flip ? (kingSq ^ 56) : kingSq;

    int p = getTypeIndex(piece.type);
    if (p >= 5) return -1; // exclude kings
    if (piece.color == Color.BLACK) p += 5;

    // (PieceIndex * 64) + Square + (KingSquare * 640)
    // Max: (9 * 64) + 63 + (63 * 640) = 40959
    final finalIdx = (p * 64) + s + (k * 640);
    if (finalIdx < 0 || finalIdx >= NUM_FEATURES) return -1;
    return finalIdx;
  }

  void _refreshAccumulator(NnueAccumulator acc, List<Piece?> board) {
    // find kings in 0x88, convert to 0..63, enumerate all valid 0x88 squares
    final wK0x88 = board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    final bK0x88 = board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );
    if (wK0x88 == -1 || bK0x88 == -1) return;

    final wK = _to64(wK0x88);
    final bK = _to64(bK0x88);

    final wf = <int>[];
    final bf = <int>[];

    for (int sq = 0; sq < 128; ++sq) {
      if ((sq & 0x88) != 0) continue;
      final p = board[sq];
      if (p == null || p.type == PieceType.KING) continue; // skip kings

      final s64 = _to64(sq);
      final wIdx = getHalfKPIndex(wK, s64, p, false);
      final bIdx = getHalfKPIndex(bK, s64, p, true);
      if (wIdx != -1) wf.add(wIdx);
      if (bIdx != -1) bf.add(bIdx);
    }

    acc.refresh(l0, wf, 0);
    acc.refresh(l0, bf, 1);
  }

  @pragma('vm:prefer-inline')
  Float64List _forwardDense(LinearLayer layer, Float64List input) {
    final out = Float64List.fromList(layer.bias);
    for (var i = 0; i < layer.numInputs; i++) {
      final val = input[i];
      if (val == 0) continue;
      final w = layer.weights[i];
      for (var j = 0; j < layer.numOutputs; j++) {
        out[j] += val * w[j];
      }
    }
    return out;
  }

  double evaluate(NnueAccumulator acc, Color sideToMove) {
    final stmIdx = sideToMove == Color.WHITE ? 0 : 1;

    final inL1 = Float64List(2 * M);
    inL1.setRange(0, M, acc.v[stmIdx]);
    inL1.setRange(M, 2 * M, acc.v[1 - stmIdx]);

    final act1 = Float64List(2 * M);
    for (int i = 0; i < 2 * M; i++) {
      act1[i] = _crelu(inL1[i]);
    }

    final out1 = _forwardDense(l1, act1);
    final act2 = Float64List(K);
    for (int i = 0; i < K; i++) {
      act2[i] = _crelu(out1[i]);
    }

    final score = _forwardDense(l2, act2)[0] * SCALE;

    // Optional periodic signal logging if desired:
    // if (t > 0 && t % 10 == 0) {
    //   int c1 = act1.where((x) => x > 0).length;
    //   int c2 = act2.where((x) => x > 0).length;
    //   print("Signal -> L1: $c1 | L2: $c2 | Score: ${score.toStringAsFixed(2)} cp");
    // }
    return score;
  }
}
