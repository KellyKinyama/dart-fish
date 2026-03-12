import 'dart:math';
import 'dart:typed_data';
import 'chess3.dart';
// --- CHESS FOUNDATION (REPLACING chess3.dart FOR RUNNABILITY) ---

// enum Color { WHITE, BLACK }

// enum PieceType {
//   PAWN(0), KNIGHT(1), BISHOP(2), ROOK(3), QUEEN(4), KING(5);
//   final int shift;
//   const PieceType(this.shift);
// }

// class Piece {
//   final PieceType type;
//   final Color color;
//   Piece(this.type, this.color);
// }

// --- NNUE ARCHITECTURE ---

class LinearLayer {
  final int numInputs;
  final int numOutputs;
  final List<Float64List> weights;
  final Float64List bias;

  LinearLayer(this.numInputs, this.numOutputs)
    : weights = List.generate(numInputs, (_) => Float64List(numOutputs)),
      bias = Float64List(numOutputs);

  void randomize(Random rng) {
    double scale = sqrt(2.0 / numInputs); // Xavier-ish initialization
    for (var i = 0; i < numInputs; i++) {
      for (var j = 0; j < numOutputs; j++) {
        weights[i][j] = (rng.nextDouble() * 2 - 1) * scale;
      }
    }
  }
}

class NnueAccumulator {
  final List<Float64List> v;
  final int m;

  NnueAccumulator(this.m) : v = [Float64List(m), Float64List(m)];

  void refresh(LinearLayer l0, List<int> active, int p) {
    v[p].setAll(0, l0.bias);
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

  double _creluDeriv(double x) => (x > 0.0 && x < 1.0) ? 1.0 : 0.0;

  /// Main Training Logic (Manual Autograd Math)
  void train(ChessEngine engine, double target, double lr) {
    final acc = engine.currentAcc;
    final sideToMove = engine.turn;

    // --- 1. FORWARD PASS ---
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

    // --- 2. BACKWARD PASS ---
    double gradOut = 2.0 * (prediction - target);

    // Gradients for L2
    final gradL2Out = Float64List(K);
    for (int i = 0; i < K; i++) {
      double dActivation = _creluDeriv(l1Out[i]);
      gradL2Out[i] = gradOut * l2.weights[i][0] * dActivation;
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
      gradL1Out[i] = sum * _creluDeriv(inputL1[i]);

      for (int j = 0; j < K; j++) {
        l1.weights[i][j] -= lr * gradL2Out[j] * l1Activated[i];
      }
    }
    for (int j = 0; j < K; j++) l1.bias[j] -= lr * gradL2Out[j];

    // --- 3. SPARSE UPDATE FOR L0 ---
    _updateL0(engine, gradL1Out, lr, sideToMove);
  }

  void _updateL0(
    ChessEngine engine,
    Float64List gradL1Out,
    double lr,
    Color stm,
  ) {
    int wKing = engine.board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    int bKing = engine.board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );
    bool isWhiteStm = stm == Color.WHITE;

    for (int i = 0; i < 64; i++) {
      final p = engine.board[i];
      if (p == null || p.type == PieceType.KING) continue;

      int stmFeat = isWhiteStm
          ? _getHalfKPIndex(wKing, i, p, false)
          : _getHalfKPIndex(bKing, i, p, true);
      int nStmFeat = isWhiteStm
          ? _getHalfKPIndex(bKing, i, p, true)
          : _getHalfKPIndex(wKing, i, p, false);

      for (int j = 0; j < M; j++) {
        l0.weights[stmFeat][j] -= lr * gradL1Out[j];
        l0.weights[nStmFeat][j] -= lr * gradL1Out[j + M];
      }
    }
  }

  int _getHalfKPIndex(int kingSq, int pieceSq, Piece piece, bool flip) {
    int sq = flip ? pieceSq ^ 56 : pieceSq;
    int kSq = flip ? kingSq ^ 56 : kingSq;
    int pIdx =
        piece.type.shift * 2 +
        (flip
            ? (piece.color == Color.WHITE ? 1 : 0)
            : (piece.color == Color.WHITE ? 0 : 1));
    return sq + (pIdx + kSq * 10) * 64;
  }

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

  double evaluate(NnueAccumulator acc, Color sideToMove) {
    final stmIdx = sideToMove == Color.WHITE ? 0 : 1;
    final inputL1 = Float64List(2 * M);
    inputL1.setRange(0, M, acc.v[stmIdx]);
    inputL1.setRange(M, 2 * M, acc.v[1 - stmIdx]);

    final l1Activated = Float64List(2 * M);
    for (int i = 0; i < 2 * M; i++) l1Activated[i] = inputL1[i].clamp(0.0, 1.0);

    final l1Out = _forwardDense(l1, l1Activated);
    final l2Activated = Float64List(K);
    for (int i = 0; i < K; i++) l2Activated[i] = l1Out[i].clamp(0.0, 1.0);

    return _forwardDense(l2, l2Activated)[0];
  }
}

// --- ENGINE ---

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
    board[4] = Piece(PieceType.KING, Color.WHITE);
    board[60] = Piece(PieceType.KING, Color.BLACK);
    board[12] = Piece(PieceType.PAWN, Color.WHITE);
  }

  void _fullRefresh() {
    int wKing = board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    int bKing = board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );

    List<int> wFeatures = [];
    List<int> bFeatures = [];

    for (int i = 0; i < 64; i++) {
      final p = board[i];
      if (p == null || p.type == PieceType.KING) continue;
      wFeatures.add(nnue._getHalfKPIndex(wKing, i, p, false));
      bFeatures.add(nnue._getHalfKPIndex(bKing, i, p, true));
    }

    currentAcc.refresh(nnue.l0, wFeatures, 0);
    currentAcc.refresh(nnue.l0, bFeatures, 1);
  }

  double getEvaluation() => nnue.evaluate(currentAcc, turn);
}

void main() {
  final engine = ChessEngine();

  print("Pre-training Eval: ${engine.getEvaluation().toStringAsFixed(6)}");

  // Training Simulation: Assume the current position is actually much better for White (+1.0)
  double targetValue = 1.0;
  print("Training toward target: $targetValue...");

  for (int epoch = 0; epoch < 50; epoch++) {
    engine.nnue.train(engine, targetValue, 0.01);
    if (epoch % 10 == 0) {
      print("Epoch $epoch Eval: ${engine.getEvaluation().toStringAsFixed(6)}");
    }
  }

  print("Post-training Eval: ${engine.getEvaluation().toStringAsFixed(6)}");
}
