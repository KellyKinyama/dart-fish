import 'dart:math';
import 'dart:typed_data';
import 'chess3.dart';

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
    double scale = sqrt(2.0 / numInputs);
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

  // --- ADAM STATE BUFFERS ---
  // L0 buffers: massive sparse memory
  late List<Float64List> m0, v0;
  // L1 buffers: flattened for speed
  late Float64List m1, v1, mb1, vb1;
  // L2 buffers
  late Float64List m2, v2, mb2, vb2;

  int t = 0; // Timestep for bias correction

  NNUE() {
    final rng = Random(42);
    l0 = LinearLayer(NUM_FEATURES, M)..randomize(rng);
    l1 = LinearLayer(2 * M, K)..randomize(rng);
    l2 = LinearLayer(K, 1)..randomize(rng);

    // Initialize Adam Moments
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

  double _creluDeriv(double x) => (x > 0.0 && x < 1.0) ? 1.0 : 0.0;

  void train(ChessEngine engine, double target, double lr) {
    t++;
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
    final double bc1 = 1.0 - pow(beta1, t);
    final double bc2 = 1.0 - pow(beta2, t);

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

    // --- 2. BACKWARD PASS (GRADIENT CALCULATION) ---
    double gradOut = 2.0 * (prediction - target);

    final gradL2Out = Float64List(K);
    for (int i = 0; i < K; i++) {
      gradL2Out[i] = gradOut * l2.weights[i][0] * _creluDeriv(l1Out[i]);
    }

    final gradL1Out = Float64List(2 * M);
    for (int i = 0; i < 2 * M; i++) {
      double sum = 0;
      for (int j = 0; j < K; j++) {
        sum += gradL2Out[j] * l1.weights[i][j];
      }
      gradL1Out[i] = sum * _creluDeriv(inputL1[i]);
    }

    // --- 3. WEIGHT UPDATES (ADAM MATH) ---

    // Update L2
    for (int i = 0; i < K; i++) {
      double g = gradOut * l2Activated[i];
      m2[i] = beta1 * m2[i] + (1.0 - beta1) * g;
      v2[i] = beta2 * v2[i] + (1.0 - beta2) * (g * g);
      l2.weights[i][0] -= lr * (m2[i] / bc1) / (sqrt(v2[i] / bc2) + epsilon);
    }
    mb2[0] = beta1 * mb2[0] + (1.0 - beta1) * gradOut;
    vb2[0] = beta2 * vb2[0] + (1.0 - beta2) * (gradOut * gradOut);
    l2.bias[0] -= lr * (mb2[0] / bc1) / (sqrt(vb2[0] / bc2) + epsilon);

    // Update L1
    for (int i = 0; i < 2 * M; i++) {
      for (int j = 0; j < K; j++) {
        int idx = i * K + j;
        double g = gradL2Out[j] * l1Activated[i];
        m1[idx] = beta1 * m1[idx] + (1.0 - beta1) * g;
        v1[idx] = beta2 * v1[idx] + (1.0 - beta2) * (g * g);
        l1.weights[i][j] -=
            lr * (m1[idx] / bc1) / (sqrt(v1[idx] / bc2) + epsilon);
      }
    }
    for (int j = 0; j < K; j++) {
      mb1[j] = beta1 * mb1[j] + (1.0 - beta1) * gradL2Out[j];
      vb1[j] = beta2 * vb1[j] + (1.0 - beta2) * (gradL2Out[j] * gradL2Out[j]);
      l1.bias[j] -= lr * (mb1[j] / bc1) / (sqrt(vb1[j] / bc2) + epsilon);
    }

    // Update L0 (Sparse Adam)
    _updateL0Adam(engine, gradL1Out, lr, bc1, bc2, beta1, beta2, epsilon);
  }

  void _updateL0Adam(
    ChessEngine engine,
    Float64List gradL1Out,
    double lr,
    double bc1,
    double bc2,
    double b1,
    double b2,
    double eps,
  ) {
    int wKing = engine.board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
    );
    int bKing = engine.board.indexWhere(
      (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
    );
    bool isWhiteStm = engine.turn == Color.WHITE;

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
        // STM Update
        double gS = gradL1Out[j];
        m0[stmFeat][j] = b1 * m0[stmFeat][j] + (1.0 - b1) * gS;
        v0[stmFeat][j] = b2 * v0[stmFeat][j] + (1.0 - b2) * (gS * gS);
        l0.weights[stmFeat][j] -=
            lr * (m0[stmFeat][j] / bc1) / (sqrt(v0[stmFeat][j] / bc2) + eps);

        // nSTM Update
        double gN = gradL1Out[j + M];
        m0[nStmFeat][j] = b1 * m0[nStmFeat][j] + (1.0 - b1) * gN;
        v0[nStmFeat][j] = b2 * v0[nStmFeat][j] + (1.0 - b2) * (gN * gN);
        l0.weights[nStmFeat][j] -=
            lr * (m0[nStmFeat][j] / bc1) / (sqrt(v0[nStmFeat][j] / bc2) + eps);
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

  double targetValue = 1.0;
  print("Training toward target: $targetValue (using Adam)...");

  for (int epoch = 0; epoch < 50; epoch++) {
    // Adam lr is usually smaller than SGD (0.001 is standard)
    engine.nnue.train(engine, targetValue, 0.001);
    if (epoch % 10 == 0) {
      print("Epoch $epoch Eval: ${engine.getEvaluation().toStringAsFixed(6)}");
    }
  }
  print("Post-training Eval: ${engine.getEvaluation().toStringAsFixed(6)}");
}
