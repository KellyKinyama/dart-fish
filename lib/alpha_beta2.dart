import 'dart:math';
import 'chess3.dart';
import 'chess_nnue.dart';

class SearchResult {
  Move? move;
  double score;
  int nodes;
  List<Move> pv; // Principal Variation: The best sequence of moves found

  SearchResult(this.move, this.score, {this.nodes = 0, this.pv = const []});
}

int nodesEvaluated = 0;

SearchResult alphaBeta(
  ChessWithNNUE game,
  int depth,
  double alpha,
  double beta,
  bool verify, [
  int maxDepth = 0,
]) {
  nodesEvaluated++;

  if (verify) _performConsistencyCheck(game);

  if (depth <= 0) {
    return SearchResult(null, game.nnueEvaluation, nodes: 1);
  }

  final moves = game.generate_moves();

  if (moves.isEmpty) {
    if (game.in_check) return SearchResult(null, -999.0 - depth, nodes: 1);
    return SearchResult(null, 0.0, nodes: 1);
  }

  Move? bestMove;
  List<Move> bestPv = [];

  for (final move in moves) {
    // FIX: Generate the SAN string BEFORE making the move
    // This prevents the Null Check error during move_to_san calls
    // String moveSan = game.move_to_san(move);

    game.make_move(move);
    var res = alphaBeta(game, depth - 1, -beta, -alpha, verify, maxDepth);
    double score = -res.score;
    game.undo_move();

    if (score >= beta) {
      return SearchResult(move, beta, nodes: nodesEvaluated);
    }

    if (score > alpha) {
      alpha = score;
      bestMove = move;
      bestPv = [move, ...res.pv];

      // Print "root" level debug info safely
      if (depth == maxDepth && bestMove != null) {
        // We use a simplified coordinate print (e.g., e2e4)
        // to avoid the move_to_san null-pointer bug
        String moveStr = "${bestMove.fromAlgebraic}${bestMove.toAlgebraic}";
        print(
          "  depth $depth | score: ${alpha.toStringAsFixed(4)} | nodes: $nodesEvaluated | best_move: $moveStr",
        );
      }
    }
  }

  return SearchResult(bestMove, alpha, nodes: nodesEvaluated, pv: bestPv);
}

void _performConsistencyCheck(ChessWithNNUE game) {
  double incrementalScore = game.nnueEvaluation;
  var savedAcc = game.accumulator.copy();
  game.fullNnueRefresh();
  double absoluteScore = game.nnueEvaluation;
  game.accumulator = savedAcc;

  double drift = (incrementalScore - absoluteScore).abs();
  if (drift > 0.0001) {
    print("\n[!] SYNC ERROR DETECTED");
    print("    FEN: ${game.fen}");
    print("    Inc: $incrementalScore | Abs: $absoluteScore | Diff: $drift");
  }
}

void main() {
  print("--- Starting Enhanced Engine Debug ---");
  final game = ChessWithNNUE();
  final stopwatch = Stopwatch()..start();

  int targetDepth = 3;
  print("\nAnalyzing position at depth $targetDepth...");

  nodesEvaluated = 0;
  var result = alphaBeta(game, targetDepth, -1000.0, 1000.0, true, targetDepth);

  stopwatch.stop();
  double seconds = max(stopwatch.elapsedMilliseconds / 1000.0, 0.001);
  double nps = nodesEvaluated / seconds;

  print("-" * 40);
  print("Best Move : ${game.move_to_san(result.move!)}");
  print("Best Line : ${result.pv.map((m) => game.move_to_san(m)).join(' ')}");
  print("Final Eval: ${result.score.toStringAsFixed(4)}");
  print("Nodes     : $nodesEvaluated");
  print("Time      : ${seconds.toStringAsFixed(2)}s");
  print("Speed     : ${nps.toStringAsFixed(0)} Nodes Per Second");
  print("-" * 40);

  // Consistency check during a simulated game
  print("\nVerifying incremental updates during sequence...");
  List<String> testMoves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "O-O"];
  for (var m in testMoves) {
    game.move(m);
    _performConsistencyCheck(game);
    print("Move $m -> ${game.nnueEvaluation.toStringAsFixed(4)} [OK]");
  }
}
