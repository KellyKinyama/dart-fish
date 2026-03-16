import 'alphabeta_train2.dart';
import 'chess3.dart';
import 'chess_nnue3.dart'; // Provides ChessWithNNUE
import 'nnue_persistence2.dart'; // Provides NNUESerializer.load/save

final ChessWithNNUE game = ChessWithNNUE();
int nodesEvaluated = 0;

/// Simple MVV-LVA score: prefer captures of higher-value pieces.
/// Uses PieceType.shift as provided by your engine: p=0,n=1,b=2,r=3,q=4,k=5
/// We weigh it for ordering only; not a static eval.
int _mvvLvaScore(Move m) {
  if (m.captured == null) return 0;
  // Give higher weight to more valuable captures.
  // Use (victim << 4) - attacker to prefer higher victim and lower attacker.
  final victim = m.captured!.shift;
  final attacker = m.piece.shift;
  return (victim << 4) - attacker;
}

/// Runs a fixed-depth negamax alpha-beta search with simple MVV-LVA ordering.
SearchResult search(int depth, {required Map<Move, RootMove> rootMoves}) {
  const double _negInf = -100000.0;
  const double _posInf = 100000.0;
  nodesEvaluated = 0;

  if (rootMoves.isEmpty) {
    // throw ArgumentError("Rootmoves cannot be empty");

    final moves = game.generate_moves();
    for (Move move in moves) {
      rootMoves[move] = RootMove(move, _negInf);
    }
  }

  // Map<Move, RootMove> rootMoves = {};

  // final moves = game.generate_moves();
  // for (Move move in moves) {
  //   rootMoves[move] = RootMove(move, 0);
  // }

  return _alphaBeta(depth, _negInf, _posInf, 0, rootMoves: rootMoves);
}

SearchResult _alphaBeta(
  int depth,
  double alpha,
  double beta,
  int ply, {
  Map<Move, RootMove> rootMoves = const {},
}) {
  nodesEvaluated++;

  // Leaf: NNUE evaluation from side to move’s perspective
  if (depth <= 0) {
    return SearchResult(null, game.nnueEvaluation, nodes: 1, pv: const []);
  }

  List<Move> moves;

  // --- Move ordering: MVV-LVA first (simple, effective baseline) ---
  if (ply == 0 && rootMoves.isNotEmpty) {
    rootMoves.values.toList().sort((a, b) => a.score.compareTo(b.score));
    // print("Rootmoves: $rootMoves");
    moves = rootMoves.keys.toList();
  } else {
    moves = game.generate_moves();
    moves.sort((a, b) => _mvvLvaScore(b).compareTo(_mvvLvaScore(a)));

    // if (ply == 0) {
    //   for (var move in moves) {
    //     rootMoves[move] = RootMove(move, alpha);
    //   }
    // }
  }
  if (moves.isEmpty) {
    // checkmate or stalemate
    if (game.in_check) {
      // Mate scores typically scaled with remaining depth (more urgent mate = higher magnitude)
      // Use -9999 + ply to avoid horizon weirdness; keep your original shape:
      return SearchResult(null, -999.0 - depth, nodes: 1, pv: const []);
    }
    return SearchResult(null, 0.0, nodes: 1, pv: const []);
  }

  Move? bestMove;
  List<Move> bestPv = const [];

  var localAlpha = alpha;

  for (final move in moves) {
    game.make_move(move);

    // Negamax: score from opponent’s point of view, then negate.
    final child = _alphaBeta(depth - 1, -beta, -localAlpha, ply + 1);
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

      if (ply == 0) {
        // print("found move: $move");
        rootMoves[move] = RootMove(move, score);
      }

      bestPv = [move, ...child.pv];
    }
  }
  if (ply == 0 && rootMoves.isEmpty) throw Exception("Rootmoves is empty");

  return SearchResult(
    bestMove,
    localAlpha,
    nodes: nodesEvaluated,
    pv: bestPv,
    rootMoves: rootMoves,
  );
}
// -----------------------------
// Demo / Test harness
// -----------------------------

Future<void> main() async {
  final game = ChessWithNNUE();
  final trainer = NNUETrainer(game);
  // final searcher = NNUESearcher(game);

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
  final initialRes = search(3, rootMoves: {});
  print(
    "Initial Eval: ${initialRes.score.toStringAsFixed(4)} cp | nodes: ${initialRes.nodes}",
  );

  // Training loop
  const int iterations = 6;
  const int samplesPerIter = 64;
  const int labelDepth = 2;
  const double lr = 0.001;

  SearchResult? searchRes = null;

  for (int iteration = 1; iteration <= iterations; iteration++) {
    print("\n--- ITERATION $iteration ---");

    trainer.generateData(samplesPerIter, labelDepth);
    trainer.runEpoch(lr);

    // Check progress from starting position
    game.reset();
    final progressRes = search(
      iteration,
      rootMoves: searchRes != null ? searchRes.rootMoves : {},
    );
    searchRes = progressRes;
    print(
      "Post-Train Eval (Root): ${progressRes.score.toStringAsFixed(4)} cp | nodes: ${progressRes.nodes}",
    );
    print("Rootmoves: ${progressRes.rootMoves}");
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
