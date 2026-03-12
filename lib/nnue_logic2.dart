import 'dart:math';
import 'dart:typed_data';

import 'chess3.dart';

// --- CHESS CONSTANTS & PIECES ---
// enum Color { white, black }

// enum PieceType { pawn, knight, bishop, rook, queen, king }

// class Piece {
//   final PieceType type;
//   final Color color;
//   Piece(this.type, this.color);
// }

// --- NNUE ARCHITECTURE ---

/// Represents a Linear (Fully Connected) Layer.
/// In NNUE, the first layer (L0) is sparse, others are dense.
class LinearLayer {
  final int numInputs;
  final int numOutputs;
  final List<Float64List> weights; // Column-major for sparse access
  final Float64List bias;

  LinearLayer(this.numInputs, this.numOutputs)
    : weights = List.generate(numInputs, (_) => Float64List(numOutputs)),
      bias = Float64List(numOutputs);

  // Helper to randomize weights (simulating a "trained" net for this demo)
  void randomize(Random rng) {
    for (var i = 0; i < numInputs; i++) {
      for (var j = 0; j < numOutputs; j++) {
        weights[i][j] = (rng.nextDouble() - 0.5) * 0.1;
      }
    }
    for (var i = 0; i < numOutputs; i++) {
      bias[i] = (rng.nextDouble() - 0.5) * 0.1;
    }
  }
}

/// The Accumulator stores the transformed features for both perspectives.
class NnueAccumulator {
  // Use a list of two Float64Lists: v[0] for White, v[1] for Black
  final List<Float64List> v;
  final int m;

  NnueAccumulator(this.m) : v = [Float64List(m), Float64List(m)];

  // The copy method correctly duplicates the data
  NnueAccumulator copy() {
    final newAcc = NnueAccumulator(m);
    newAcc.v[0].setAll(0, v[0]);
    newAcc.v[1].setAll(0, v[1]);
    return newAcc;
  }

  /// Incremental update: Only process the features that changed.
  /// [p] is 0 for White perspective, 1 for Black.
  void update(LinearLayer l0, List<int> added, List<int> removed, int p) {
    for (var idx in removed) {
      for (var i = 0; i < m; i++) {
        v[p][i] -= l0.weights[idx][i];
      }
    }
    for (var idx in added) {
      for (var i = 0; i < m; i++) {
        v[p][i] += l0.weights[idx][i];
      }
    }
  }

  /// Full refresh from a list of active features
  void refresh(LinearLayer l0, List<int> active, int p) {
    v[p].setAll(0, l0.bias); // Initialize with bias
    for (var idx in active) {
      for (var i = 0; i < m; i++) {
        v[p][i] += l0.weights[idx][i];
      }
    }
  }
}

class NNUE {
  static const int M = 256;
  static const int K = 32;
  static const int NUM_FEATURES = 40960;

  late LinearLayer l0, l1, l2;

  NNUE() {
    final rng = Random(42);
    l0 = LinearLayer(NUM_FEATURES, M)..randomize(rng);
    l1 = LinearLayer(2 * M, K)..randomize(rng);
    l2 = LinearLayer(K, 1)..randomize(rng);
  }

  /// Helper: Derivative of Clipped ReLU
  double _creluDeriv(double x) => (x > 0.0 && x < 1.0) ? 1.0 : 0.0;

  void train(NnueAccumulator acc, Color sideToMove, double target, double lr) {
    // --- 1. FORWARD PASS (Keep track of intermediate values) ---
    final stmIdx = sideToMove == Color.WHITE ? 0 : 1;
    final inputL1 = Float64List(2 * M);
    inputL1.setRange(0, M, acc.v[stmIdx]);
    inputL1.setRange(M, 2 * M, acc.v[1 - stmIdx]);

    final l1Activated = Float64List(2 * M);
    for (int i = 0; i < 2 * M; i++) l1Activated[i] = inputL1[i].clamp(0.0, 1.0);

    final l1Out = _forwardDense(l1, l1Activated);
    final l2Activated = Float64List(K);
    for (int i = 0; i < K; i++) l2Activated[i] = l1Out[i].clamp(0.0, 1.0);

    final l2Out = _forwardDense(l2, l2Activated);
    double prediction = l2Out[0];

    // --- 2. BACKWARD PASS (The Math) ---

    // Gradient of Loss w.r.t Output
    double gradOut = 2.0 * (prediction - target);

    // Gradients for L2
    final gradL2Out = Float64List(K);
    for (int i = 0; i < K; i++) {
      double dActivation = _creluDeriv(l1Out[i]);
      gradL2Out[i] = gradOut * l2.weights[i][0] * dActivation;

      // Update L2 Weights & Bias
      l2.weights[i][0] -= lr * gradOut * l2Activated[i];
    }
    l2.bias[0] -= lr * gradOut;

    // Gradients for L1
    final gradL1Out = Float64List(2 * M);
    for (int i = 0; i < 2 * M; i++) {
      double sum = 0;
      for (int j = 0; j < K; j++) {
        sum += gradL2Out[j] * l1.weights[i][j];
      }
      double dActivation = _creluDeriv(inputL1[i]);
      gradL1Out[i] = sum * dActivation;

      // Update L1 Weights & Bias
      for (int j = 0; j < K; j++) {
        l1.weights[i][j] -= lr * gradL2Out[j] * l1Activated[i];
      }
    }
    for (int j = 0; j < K; j++) l1.bias[j] -= lr * gradL2Out[j];

    // --- 3. SPARSE UPDATE FOR L0 ---
    // We only update features that were actually present in this position
    List<int> activeFeatures = _getActiveFeaturesFromAccumulator(
      acc,
      sideToMove,
    );
    for (int featureIdx in activeFeatures) {
      for (int i = 0; i < M; i++) {
        // Gradient flows back through the concatenated inputL1
        l0.weights[featureIdx][i] -= lr * gradL1Out[i];
      }
    }
  }

  List<int> _getActiveFeatures(ChessEngine engine, Color sideToMove) {
    int whiteKingSq = -1;
    int blackKingSq = -1;

    // 1. Locate Kings
    for (int i = 0; i < 64; i++) {
      final p = engine.board[i];
      if (p?.type == PieceType.KING) {
        if (p!.color == Color.WHITE)
          whiteKingSq = i;
        else
          blackKingSq = i;
      }
    }

    List<int> activeFeatures = [];
    bool isWhiteStm = sideToMove == Color.WHITE;

    for (int i = 0; i < 64; i++) {
      final p = engine.board[i];
      if (p == null || p.type == PieceType.KING) continue;

      // Feature index for the perspective of the side to move (STM)
      int stmFeature = isWhiteStm
          ? _getHalfKPIndex(whiteKingSq, i, p, false)
          : _getHalfKPIndex(blackKingSq, i, p, true);

      // Feature index for the perspective of the opponent (nSTM)
      int nStmFeature = isWhiteStm
          ? _getHalfKPIndex(blackKingSq, i, p, true)
          : _getHalfKPIndex(whiteKingSq, i, p, false);

      activeFeatures.add(stmFeature);
      activeFeatures.add(nStmFeature);
    }
    return activeFeatures;
  }

  /// Generates the HalfKP index for a piece relative to the king.
  int _getHalfKPIndex(int kingSq, int pieceSq, Piece piece, bool flip) {
    int sq = flip ? pieceSq ^ 56 : pieceSq;
    int kSq = flip ? kingSq ^ 56 : kingSq;
    int pIdx =
        piece.type.shift * 2 +
        (flip
            ? (piece.color == Color.WHITE ? 1 : 0)
            : (piece.color == Color.WHITE ? 0 : 1));
    // Simplified HalfKP mapping
    return sq + (pIdx + kSq * 10) * 64;
  }

  /// Standard Dense Linear Pass
  Float64List _forwardDense(LinearLayer layer, Float64List input) {
    final out = Float64List.fromList(layer.bias);
    for (var i = 0; i < layer.numInputs; i++) {
      if (input[i] == 0) continue;
      for (var j = 0; j < layer.numOutputs; j++) {
        out[j] += input[i] * layer.weights[i][j];
      }
    }
    return out;
  }
}

// --- INTEGRATED BOARD & EVALUATION ---

class ChessEngine {
  final List<Piece?> board = List.filled(64, null);
  final NNUE nnue = NNUE();
  late NnueAccumulator currentAcc;
  Color turn = Color.WHITE;

  ChessEngine() {
    currentAcc = NnueAccumulator(NNUE.M);
    _setupStartingPosition();
    _fullRefresh();
  }

  void _setupStartingPosition() {
    // Minimal setup for demonstration: White King A1, White Pawn C3, Black King B8
    board[0] = Piece(PieceType.KING, Color.WHITE); // A1
    board[18] = Piece(PieceType.PAWN, Color.WHITE); // C3
    board[57] = Piece(PieceType.KING, Color.BLACK); // B8
  }

  /// Generates the HalfKP index for a piece relative to the king.
  int _getHalfKPIndex(int kingSq, int pieceSq, Piece piece, bool flip) {
    int sq = flip ? pieceSq ^ 56 : pieceSq;
    int kSq = flip ? kingSq ^ 56 : kingSq;
    int pIdx =
        piece.type.shift * 2 +
        (flip
            ? (piece.color == Color.WHITE ? 1 : 0)
            : (piece.color == Color.WHITE ? 0 : 1));
    // Simplified HalfKP mapping
    return sq + (pIdx + kSq * 10) * 64;
  }

  void _fullRefresh() {
    int whiteKing = board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    int blackKing = board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );

    List<int> whiteFeatures = [];
    List<int> blackFeatures = [];

    for (int i = 0; i < 64; i++) {
      final p = board[i];
      if (p == null || p.type == PieceType.KING) continue;
      whiteFeatures.add(_getHalfKPIndex(whiteKing, i, p, false));
      blackFeatures.add(_getHalfKPIndex(blackKing, i, p, true));
    }

    currentAcc.refresh(nnue.l0, whiteFeatures, 0);
    currentAcc.refresh(nnue.l0, blackFeatures, 1);
  }

  /// Simulate a move: C3 to C4
  void makeMove(int from, int to) {
    final p = board[from]!;

    // In a real engine, we'd only update the accumulator incrementally here.
    // For this example, we move the piece and trigger a refresh.
    board[to] = p;
    board[from] = null;

    _fullRefresh();
    turn = (turn == Color.WHITE) ? Color.BLACK : Color.WHITE;

    print("Move made from $from to $to. Turn is now $turn.");
  }

  double getEvaluation() {
    return nnue.evaluate(currentAcc, turn);
  }
}
