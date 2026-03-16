import 'dart:math';
import 'chess3.dart';
import 'chess_nnue3.dart'; // Provides ChessWithNNUE
import 'nnue_logic_batch2.dart'; // Provides TrainingPosition (if you re-use) and NNUE core
import 'nnue_persistence2.dart'; // Provides NNUESerializer.load/save

// -----------------------------
// Search & Training Orchestrator
// -----------------------------

class RootMove {
  Move move;
  double score;

  RootMove(this.move, this.score);

  @override
  String toString() {
    // TODO: implement toString
    return "RootMove(move: ${move}: score: ${score.toStringAsFixed(2)})";
  }
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

  /// Simple MVV-LVA score: prefer captures of higher-value pieces.
  /// Uses PieceType.shift as provided by your engine: p=0,n=1,b=2,r=3,q=4,k=5
  /// We weigh it for ordering only; not a static eval.
  static int _mvvLvaScore(Move m) {
    if (m.captured == null) return 0;
    // Give higher weight to more valuable captures.
    // Use (victim << 4) - attacker to prefer higher victim and lower attacker.
    final victim = m.captured!.shift;
    final attacker = m.piece.shift;
    return (victim << 4) - attacker;
  }

  SearchResult _alphaBeta(int depth, double alpha, double beta) {
    nodesEvaluated++;

    // Leaf: NNUE evaluation from side to move’s perspective
    if (depth <= 0) {
      return SearchResult(null, game.nnueEvaluation, nodes: 1, pv: const []);
    }

    final moves = game.generate_moves();
    if (moves.isEmpty) {
      // checkmate or stalemate
      if (game.in_check) {
        // Mate scores typically scaled with remaining depth (more urgent mate = higher magnitude)
        // Use -9999 + ply to avoid horizon weirdness; keep your original shape:
        return SearchResult(null, -999.0 - depth, nodes: 1, pv: const []);
      }
      return SearchResult(null, 0.0, nodes: 1, pv: const []);
    }

    // --- Move ordering: MVV-LVA first (simple, effective baseline) ---
    moves.sort((a, b) => _mvvLvaScore(b).compareTo(_mvvLvaScore(a)));

    Move? bestMove;
    List<Move> bestPv = const [];

    var localAlpha = alpha;

    for (final move in moves) {
      game.make_move(move);

      // Negamax: score from opponent’s point of view, then negate.
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

class NNUETrainer {
  final ChessWithNNUE game;
  final List<TrainingPosition> trainingBuffer = [];
  final int batchSize;

  final Random _rng;

  NNUETrainer(this.game, {this.batchSize = 32, Random? rng})
    : _rng = rng ?? Random();

  /// Deep-copy board so stored samples are immutable (Pieces are mutable in undo logic).
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

      // 2) Store a deep-copied board + stm + target score
      trainingBuffer.add(
        TrainingPosition(_cloneBoard(game.board), game.turn, res.score),
      );

      // 3) Diversify positions: play a random legal move or reset if none
      final moves = game.generate_moves();
      if (moves.isNotEmpty) {
        final mv = moves[_rng.nextInt(moves.length)];
        game.make_move(mv);
      } else {
        game.reset();
      }
    }
  }

  /// Trains the NNUE model using buffered samples in mini-batches.
  void runEpoch(double lr) {
    if (trainingBuffer.length < batchSize) {
      print(
        "Training buffer has ${trainingBuffer.length} < $batchSize; skipping epoch.",
      );
      return;
    }

    print("Starting training epoch on ${trainingBuffer.length} positions...");
    trainingBuffer.shuffle(_rng);

    for (int i = 0; i <= trainingBuffer.length - batchSize; i += batchSize) {
      final batch = trainingBuffer.sublist(i, i + batchSize);
      game.nnue.trainBatch(batch, lr);
    }

    trainingBuffer.clear();
    print("Training complete. Weights updated via Adam.");
  }
}

// -----------------------------
// Demo / Test harness
// -----------------------------

Future<void> main() async {
  final game = ChessWithNNUE();
  final trainer = NNUETrainer(game);
  
  final searcher = NNUESearcher(game);

  const String modelPath = 'chess_model_v2.json';

  // 1) Load weights if available
  try {
    await NNUESerializer.load(game.nnue, modelPath);
    print("Loaded NNUE model from $modelPath");
  } catch (e) {
    print("No existing model to load (or load failed): $e");
  }

  print("--- [DART-FISH] NNUE TRAINING SYSTEM ---");

  // Baseline
  final initialRes = searcher.search(3);
  print(
    "Initial Eval: ${initialRes.score.toStringAsFixed(4)} cp | nodes: ${initialRes.nodes}",
  );

  // Training loop
  const int iterations = 5;
  const int samplesPerIter = 64;
  const int labelDepth = 2;
  const double lr = 0.001;

  for (int iteration = 1; iteration <= iterations; iteration++) {
    print("\n--- ITERATION $iteration ---");

    trainer.generateData(samplesPerIter, labelDepth);
    trainer.runEpoch(lr);

    // Check progress from starting position
    game.reset();
    final progressRes = searcher.search(3);
    print(
      "Post-Train Eval (Root): ${progressRes.score.toStringAsFixed(4)} cp | nodes: ${progressRes.nodes}",
    );
  }

  print("\nFinal Test Complete. Engine has 'learned' from search data.");

  // 3) Save progress (FIXED: pass `game.nnue`, not `game`)
  try {
    await NNUESerializer.save(game.nnue, modelPath);
    print("Saved NNUE model to $modelPath");
  } catch (e) {
    print("Failed to save model: $e");
  }
}
