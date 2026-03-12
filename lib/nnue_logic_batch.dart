import 'dart:math';
import 'dart:typed_data';
import 'chess3.dart';

class TrainingPosition {
  final List<Piece?> board;
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
    double scale = sqrt(2.0 / numInputs) * 1.1;
    for (var i = 0; i < numInputs; i++) {
      for (var j = 0; j < numOutputs; j++) {
        weights[i][j] = (rng.nextDouble() * 2 - 1) * scale;
      }
    }
    for (var i = 0; i < numOutputs; i++) bias[i] = 0.02;
  }
}

class NnueAccumulator {
  final List<Float64List> v;
  final int m;
  NnueAccumulator(this.m) : v = [Float64List(m), Float64List(m)];

  void refresh(LinearLayer l0, List<int> active, int p) {
    v[p].setAll(0, l0.bias);
    for (var idx in active) {
      if (idx >= 0 && idx < l0.numInputs) {
        final w = l0.weights[idx];
        for (var i = 0; i < m; i++) v[p][i] += w[i];
      }
    }
  }

  void update(LinearLayer l0, List<int> added, List<int> removed, int p) {
    for (var idx in removed) {
      if (idx >= 0 && idx < l0.numInputs) {
        final w = l0.weights[idx];
        for (var i = 0; i < m; i++) v[p][i] -= w[i];
      }
    }
    for (var idx in added) {
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
  static const int M = 256;
  static const int K = 32;
  static const int NUM_FEATURES = 41024;
  static const double SCALE = 400.0;

  late LinearLayer l0, l1, l2;
  late List<Float64List> m0, v0;
  late Float64List m1, v1, mb1, vb1, m2, v2, mb2, vb2;
  int t = 0;

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
    m1 = Float64List((2 * M) * K);
    v1 = Float64List((2 * M) * K);
    mb1 = Float64List(K);
    vb1 = Float64List(K);
    m2 = Float64List(K);
    v2 = Float64List(K);
    mb2 = Float64List(1);
    vb2 = Float64List(1);
  }

  double _crelu(double x) => x.clamp(0.0, 1.0);
  double _creluDeriv(double x) => (x > 0.0 && x < 1.0) ? 1.0 : 0.0;

  int getTypeIndex(PieceType type) {
    if (type == PieceType.PAWN) return 0;
    if (type == PieceType.KNIGHT) return 1;
    if (type == PieceType.BISHOP) return 2;
    if (type == PieceType.ROOK) return 3;
    if (type == PieceType.QUEEN) return 4;
    return 5; // King
  }

  // int getHalfKPIndex(int kingSq, int pieceSq, Piece piece, bool flip) {
  //   int s = flip ? pieceSq ^ 56 : pieceSq;
  //   int k = flip ? kingSq ^ 56 : kingSq;
  //   int p = getTypeIndex(piece.type);
  //   if (p == 5) return -1; // Skip Kings in the feature mapping
  //   if (piece.color == Color.BLACK) p += 5;

  //   // (Piece * 64 squares) + Square + (KingSquare * 640)
  //   // Range: 0 to 40959 (Fits in 41024)
  //   return (p * 64) + s + (k * 640);
  // }

  void trainBatch(List<TrainingPosition> batch, double lr) {
    if (batch.isEmpty) return;
    t++;
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    final double bc1 = 1.0 - pow(b1, t);
    final double bc2 = 1.0 - pow(b2, t);

    var gL2W = Float64List(K);
    var gL2B = 0.0;
    var gL1W = Float64List((2 * M) * K);
    var gL1B = Float64List(K);
    var gL0W = <int, Float64List>{};

    for (var pos in batch) {
      final acc = NnueAccumulator(M);
      _refreshAccumulator(acc, pos.board);

      final stmIdx = pos.turn == Color.WHITE ? 0 : 1;
      final inL1 = Float64List(2 * M);
      inL1.setRange(0, M, acc.v[stmIdx]);
      inL1.setRange(M, 2 * M, acc.v[1 - stmIdx]);

      final act1 = Float64List(2 * M);
      for (int i = 0; i < 2 * M; i++) act1[i] = _crelu(inL1[i]);

      final out1 = _forwardDense(l1, act1);
      final act2 = Float64List(K);
      for (int i = 0; i < K; i++) act2[i] = _crelu(out1[i]);

      double pred = _forwardDense(l2, act2)[0];
      double gradOut = 2.0 * (pred - (pos.target / SCALE)) / batch.length;

      // Backprop Layer 2
      final gL2Out = Float64List(K);
      for (int i = 0; i < K; i++) {
        gL2Out[i] = gradOut * l2.weights[i][0] * _creluDeriv(out1[i]);
        gL2W[i] += gradOut * act2[i];
      }
      gL2B += gradOut;

      // Backprop Layer 1
      final gL1Out = Float64List(2 * M);
      for (int i = 0; i < 2 * M; i++) {
        double sum = 0;
        for (int j = 0; j < K; j++) {
          sum += gL2Out[j] * l1.weights[i][j];
          gL1W[i * K + j] += gL2Out[j] * act1[i];
        }
        gL1Out[i] = sum * _creluDeriv(inL1[i]);
      }
      for (int j = 0; j < K; j++) gL1B[j] += gL2Out[j];

      // Backprop Layer 0 (Accumulator)
      _accumulateL0(pos, gL1Out, gL0W);
    }

    _applyAdamUpdates(lr, b1, b2, bc1, bc2, eps, gL2W, gL2B, gL1W, gL1B, gL0W);
  }

  void _accumulateL0(
    TrainingPosition pos,
    Float64List gL1Out,
    Map<int, Float64List> gL0W,
  ) {
    int wK = pos.board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    int bK = pos.board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );
    if (wK == -1 || bK == -1) return;

    for (int i = 0; i < 64; i++) {
      final p = pos.board[i];
      if (p == null) continue;

      int wf = getHalfKPIndex(wK, i, p, false);
      int bf = getHalfKPIndex(bK, i, p, true);

      if (wf != -1) {
        gL0W.putIfAbsent(wf, () => Float64List(M));
        for (int j = 0; j < M; j++) gL0W[wf]![j] += gL1Out[j];
      }
      if (bf != -1) {
        gL0W.putIfAbsent(bf, () => Float64List(M));
        for (int j = 0; j < M; j++) gL0W[bf]![j] += gL1Out[j + M];
      }
    }
  }

  // REPLACE your _applyAdamUpdates with this version:
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
  ) {
    // Update Layer 2 (Output Layer)
    for (int i = 0; i < K; i++) {
      m2[i] = b1 * m2[i] + (1.0 - b1) * gL2W[i];
      v2[i] = b2 * v2[i] + (1.0 - b2) * (gL2W[i] * gL2W[i]);
      l2.weights[i][0] -= lr * (m2[i] / bc1) / (sqrt(v2[i] / bc2) + eps);
    }
    mb2[0] = b1 * mb2[0] + (1.0 - b1) * gL2B;
    vb2[0] = b2 * vb2[0] + (1.0 - b2) * (gL2B * gL2B);
    l2.bias[0] -= lr * (mb2[0] / bc1) / (sqrt(vb2[0] / bc2) + eps);

    // Update Layer 1 (Hidden Layer)
    for (int i = 0; i < (2 * M) * K; i++) {
      m1[i] = b1 * m1[i] + (1.0 - b1) * gL1W[i];
      v1[i] = b2 * v1[i] + (1.0 - b2) * (gL1W[i] * gL1W[i]);
      int row = i ~/ K;
      int col = i % K;
      l1.weights[row][col] -= lr * (m1[i] / bc1) / (sqrt(v1[i] / bc2) + eps);
    }
    for (int j = 0; j < K; j++) {
      mb1[j] = b1 * mb1[j] + (1.0 - b1) * gL1B[j];
      vb1[j] = b2 * vb1[j] + (1.0 - b2) * (gL1B[j] * gL1B[j]);
      l1.bias[j] -= lr * (mb1[j] / bc1) / (sqrt(vb1[j] / bc2) + eps);
    }

    // Update Layer 0 (Input/Accumulator Layer)
    // CRITICAL FIX: Added bounds check inside the forEach to prevent the 74752 error
    gL0W.forEach((idx, grads) {
      if (idx >= 0 && idx < NUM_FEATURES) {
        // <--- THIS PREVENTS THE CRASH
        for (int j = 0; j < M; j++) {
          m0[idx][j] = b1 * m0[idx][j] + (1.0 - b1) * grads[j];
          v0[idx][j] = b2 * v0[idx][j] + (1.0 - b2) * (grads[j] * grads[j]);
          l0.weights[idx][j] -=
              lr * (m0[idx][j] / bc1) / (sqrt(v0[idx][j] / bc2) + eps);
        }
      }
    });
  }

  // Ensure your getHalfKPIndex looks EXACTLY like this:
  int getHalfKPIndex(int kingSq, int pieceSq, Piece piece, bool flip) {
    if (kingSq < 0 || kingSq > 63 || pieceSq < 0 || pieceSq > 63) return -1;

    int s = flip ? pieceSq ^ 56 : pieceSq;
    int k = flip ? kingSq ^ 56 : kingSq;

    int p = getTypeIndex(piece.type);
    if (p >= 5)
      return -1; // 5 is King, we don't include King in the feature map

    if (piece.color == Color.BLACK) p += 5;

    // Formula: (PieceIndex * 64) + Square + (KingSquare * 640)
    // Max: (9 * 64) + 63 + (63 * 640) = 40959
    int finalIdx = (p * 64) + s + (k * 640);

    // Final safety check
    if (finalIdx < 0 || finalIdx >= NUM_FEATURES) return -1;
    return finalIdx;
  }

  void _refreshAccumulator(NnueAccumulator acc, List<Piece?> board) {
    int wK = board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    int bK = board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );
    if (wK == -1 || bK == -1) return;

    List<int> wf = [], bf = [];
    for (int i = 0; i < 64; i++) {
      final p = board[i];
      if (p == null) continue;
      int wIdx = getHalfKPIndex(wK, i, p, false);
      int bIdx = getHalfKPIndex(bK, i, p, true);
      if (wIdx != -1) wf.add(wIdx);
      if (bIdx != -1) bf.add(bIdx);
    }
    acc.refresh(l0, wf, 0);
    acc.refresh(l0, bf, 1);
  }

  Float64List _forwardDense(LinearLayer layer, Float64List input) {
    final out = Float64List.fromList(layer.bias);
    for (var i = 0; i < layer.numInputs; i++) {
      final val = input[i];
      if (val == 0) continue;
      final w = layer.weights[i];
      for (var j = 0; j < layer.numOutputs; j++) out[j] += val * w[j];
    }
    return out;
  }

  double evaluate(NnueAccumulator acc, Color sideToMove) {
    final stmIdx = sideToMove == Color.WHITE ? 0 : 1;
    final inL1 = Float64List(2 * M);
    inL1.setRange(0, M, acc.v[stmIdx]);
    inL1.setRange(M, 2 * M, acc.v[1 - stmIdx]);

    final act1 = Float64List(2 * M);
    int c1 = 0;
    for (int i = 0; i < 2 * M; i++) {
      act1[i] = _crelu(inL1[i]);
      if (act1[i] > 0) c1++;
    }

    final out1 = _forwardDense(l1, act1);
    final act2 = Float64List(K);
    int c2 = 0;
    for (int i = 0; i < K; i++) {
      act2[i] = _crelu(out1[i]);
      if (act2[i] > 0) c2++;
    }

    double score = _forwardDense(l2, act2)[0] * SCALE;
    if (t % 10 == 0) {
      print(
        "Signal Test -> L1: $c1 | L2: $c2 | Score: ${score.toStringAsFixed(2)} CP",
      );
    }
    return score;
  }
}
