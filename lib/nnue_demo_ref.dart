import 'dart:math';
import 'chess3.dart';
import 'chess_with_nnue.dart'; // Your ChessWithNNUE wired to NNUERef
import 'nnue_logic_batch2.dart'; // TrainingPosition (board, turn, target in cp)
import 'nnue_persistence3.dart'; // Serializer that supports NNUERef

// -----------------------------
// Search & Training Orchestrator
// -----------------------------

class RootMove {
  Move move;
  double score;
  RootMove(this.move, this.score);

  @override
  String toString() =>
      "RootMove(move: $move, score: ${score.toStringAsFixed(2)})";
}

class SearchResult {
  final Move? move;
  final double score;
  final int nodes;
  final List<Move> pv;
  Map<Move, RootMove> rootMoves = {};

  SearchResult(
    this.move,
    this.score, {
    this.nodes = 0,
    List<Move>? pv,
    this.rootMoves = const {},
  }) : pv = pv ?? const <Move>[];
}

class NNUESearcher {
  // Scoring bounds (centipawns)
  static const double _negInf = -100000.0;
  static const double _posInf = 100000.0;

  final ChessWithNNUE game;
  int nodesEvaluated = 0;

  NNUESearcher(this.game);

  /// Runs a fixed-depth negamax alpha-beta search with simple MVV-LVA ordering.
  SearchResult search(int depth) {
    nodesEvaluated = 0;
    return _alphaBeta(depth, _negInf, _posInf);
  }

  /// Simple MVV-LVA: prefer captures of higher-value pieces.
  static int _mvvLvaScore(Move m) {
    if (m.captured == null) return 0;
    final victim = m.captured!.shift;
    final attacker = m.piece.shift;
    return (victim << 4) - attacker;
  }

  SearchResult _alphaBeta(int depth, double alpha, double beta) {
    nodesEvaluated++;

    // Leaf: NNUE evaluation from side-to-move perspective
    if (depth <= 0) {
      return SearchResult(null, game.nnueEvaluation, nodes: 1, pv: const []);
    }

    final moves = game.generate_moves();
    if (moves.isEmpty) {
      if (game.in_check)
        return SearchResult(null, -999.0 - depth, nodes: 1, pv: const []);
      return SearchResult(null, 0.0, nodes: 1, pv: const []);
    }

    // MVV-LVA ordering
    moves.sort((a, b) => _mvvLvaScore(b).compareTo(_mvvLvaScore(a)));

    Move? bestMove;
    List<Move> bestPv = const [];
    var localAlpha = alpha;

    for (final move in moves) {
      game.make_move(move);

      // Negamax recurse
      final child = _alphaBeta(depth - 1, -beta, -localAlpha);
      final score = -child.score;

      game.undo_move();

      // Beta cutoff
      if (score >= beta) {
        return SearchResult(
          move,
          beta,
          nodes: nodesEvaluated,
          pv: [move, ...child.pv],
        );
      }

      if (score > localAlpha) {
        localAlpha = score;
        bestMove = move;
        bestPv = [move, ...child.pv];
      }
    }

    return SearchResult(
      bestMove,
      localAlpha,
      nodes: nodesEvaluated,
      pv: bestPv,
    );
  }
}

/// Trainer that buffers positions labeled by search and calls a user-provided
/// training step (if available).
class NNUETrainer {
  final ChessWithNNUE game;
  final List<TrainingPosition> trainingBuffer = [];
  final int batchSize;
  final Random _rng;

  /// Provide a train step callback: (batch, lr) { ... }
  /// If null, training is a no-op.
  final void Function(List<TrainingPosition> batch, double lr)? trainStep;

  NNUETrainer(this.game, {this.batchSize = 32, this.trainStep, Random? rng})
    : _rng = rng ?? Random();

  /// Deep-copy board so stored samples are immutable (Pieces mutate on undo).
  static List<Piece?> _cloneBoard(List<Piece?> board) {
    final out = List<Piece?>.filled(board.length, null);
    for (int i = 0; i < board.length; i++) {
      final p = board[i];
      if (p != null) out[i] = Piece(p.type, p.color);
    }
    return out;
  }

  /// Generates training data by searching various positions and recording (board, stm, eval).
  void generateData(int numPositions, int depth) {
    assert(numPositions > 0 && depth > 0);
    print("Generating $numPositions samples using depth $depth search...");
    final searcher = NNUESearcher(game);

    for (int i = 0; i < numPositions; i++) {
      // 1) Label current position with search
      final res = searcher.search(depth);

      // 2) Store a deep-copied board + stm + target score (centipawns)
      trainingBuffer.add(
        TrainingPosition(_cloneBoard(game.board), game.turn, res.score),
      );

      // 3) Diversify: play a random legal move, or reset if none
      final moves = game.generate_moves();
      if (moves.isNotEmpty) {
        final mv = moves[_rng.nextInt(moves.length)];
        game.make_move(mv);
      } else {
        game.reset();
      }
    }
  }

  // --------- NEW: Loss helpers (MSE in cp^2) ---------

  /// Predict CP for a sample using current model/accumulator rebuild.
  double _predictCp(TrainingPosition tp) {
    final acc = game.nnue.newAccumulator();
    game.nnue.refreshAccumulator(acc, tp.board);
    return game.nnue.evaluate(acc, tp.turn);
  }

  /// Compute MSE in CP^2 for a list of samples using current model.
  double _mseOnSamplesCp(List<TrainingPosition> samples) {
    if (samples.isEmpty) return 0.0;
    double sse = 0.0;
    for (final tp in samples) {
      final pred = _predictCp(tp);
      final err = pred - tp.target; // both in CP
      sse += err * err;
    }
    return sse / samples.length;
  }

  /// Trains the NNUE model using buffered samples in mini-batches (if trainStep provided).
  /// Logs pre-/post-batch MSE (cp^2) and epoch averages.
  void runEpoch(double lr) {
    if (trainingBuffer.length < batchSize) {
      print(
        "Training buffer has ${trainingBuffer.length} < $batchSize; skipping epoch.",
      );
      return;
    }
    if (trainStep == null) {
      print(
        "No trainStep callback provided (NNUERef has no trainBatch by default). Skipping training.",
      );
      trainingBuffer.clear();
      return;
    }

    print("Starting training epoch on ${trainingBuffer.length} positions...");
    trainingBuffer.shuffle(_rng);

    double epochPre = 0.0;
    double epochPost = 0.0;
    int batches = 0;

    for (int i = 0; i <= trainingBuffer.length - batchSize; i += batchSize) {
      final batch = trainingBuffer.sublist(i, i + batchSize);

      // Pre-batch MSE
      final preMse = _mseOnSamplesCp(batch);
      print(
        "  Batch ${i ~/ batchSize + 1} pre-MSE: ${preMse.toStringAsFixed(4)} cp^2",
      );

      // Update
      trainStep!(batch, lr);

      // Post-batch MSE
      final postMse = _mseOnSamplesCp(batch);
      print(
        "  Batch ${i ~/ batchSize + 1} post-MSE: ${postMse.toStringAsFixed(4)} cp^2",
      );

      epochPre += preMse;
      epochPost += postMse;
      batches++;
    }

    if (batches > 0) {
      print(
        "Epoch avg pre-MSE: ${(epochPre / batches).toStringAsFixed(4)} cp^2 | "
        "post-MSE: ${(epochPost / batches).toStringAsFixed(4)} cp^2",
      );
    }

    trainingBuffer.clear();
    print("Training complete. Weights updated.");
  }
}

// -----------------------------
// Demo / Test harness
// -----------------------------

Future<void> main() async {
  final game = ChessWithNNUE();
  final searcher = NNUESearcher(game);

  // Enable learning (requires NNUERef.trainBatch to be implemented).
  final trainer = NNUETrainer(
    game,
    batchSize: 32,
    trainStep: (batch, lr) => game.nnue.trainBatch(batch, lr),
  );

  const String modelPath = 'chess_model_v2.json';

  // 1) Load weights if available (optional; only if your serializer supports NNUERef)
  try {
    await NNUESerializer.load(game.nnue, modelPath);
    print("Loaded NNUE model from $modelPath");
  } catch (e) {
    print("No existing model to load (or load failed): $e");
  }

  print("--- [DART-FISH] NNUE (NNUERef) TRAINING SYSTEM ---");

  // Optional: build a small validation set once
  final rng = Random(123);
  final validation = <TrainingPosition>[];
  {
    // generate 32 random samples from shallow search for validation
    const valCount = 32;
    for (int i = 0; i < valCount; i++) {
      final res = searcher.search(2);
      validation.add(
        TrainingPosition(List<Piece?>.from(game.board), game.turn, res.score),
      );
      final moves = game.generate_moves();
      if (moves.isNotEmpty) {
        game.make_move(moves[rng.nextInt(moves.length)]);
      } else {
        game.reset();
      }
    }
    // Return to start
    game.reset();
  }

  // Baseline eval
  final initialRes = searcher.search(3);
  print(
    "Initial Eval: ${initialRes.score.toStringAsFixed(4)} cp | nodes: ${initialRes.nodes}",
  );

  // Training loop
  const int iterations = 5;
  const int samplesPerIter = 64;
  const int labelDepth = 2;
  const double lr = 0.001;

  // Helper to compute validation MSE
  double _mseOnValidation() {
    double sse = 0.0;
    for (final tp in validation) {
      final acc = game.nnue.newAccumulator();
      game.nnue.refreshAccumulator(acc, tp.board);
      final pred = game.nnue.evaluate(acc, tp.turn);
      final err = pred - tp.target;
      sse += err * err;
    }
    return validation.isEmpty ? 0.0 : (sse / validation.length);
  }

  for (int iteration = 1; iteration <= iterations; iteration++) {
    print("\n--- ITERATION $iteration ---");

    trainer.generateData(samplesPerIter, labelDepth);
    trainer.runEpoch(lr);

    // Validation MSE after epoch
    final valMse = _mseOnValidation();
    print("Validation MSE: ${valMse.toStringAsFixed(4)} cp^2");

    // Check progress from starting position
    game.reset();
    final progressRes = searcher.search(3);
    print(
      "Post-Train Eval (Root): ${progressRes.score.toStringAsFixed(4)} cp | nodes: ${progressRes.nodes}",
    );
  }

  print("\nFinal Test Complete.");

  // 3) Save progress (optional; only if your serializer supports NNUERef)
  try {
    await NNUESerializer.save(game.nnue, modelPath);
    print("Saved NNUE model to $modelPath");
  } catch (e) {
    print("Failed to save model: $e");
  }
}
