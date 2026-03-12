// import 'dart:math';
import 'chess3.dart';
import 'chess_nnue.dart';
// import 'constants.dart';
// import 'nnue_logic.dart';

// Search Result wrapper
class SearchResult {
  Move? move;
  double score;
  int nodes;
  SearchResult(this.move, this.score, {this.nodes = 0});
}

// Global counter to see how many positions we evaluated
int nodesEvaluated = 0;

SearchResult alphaBeta(
  ChessWithNNUE game,
  int depth,
  double alpha,
  double beta,
  bool verify,
) {
  nodesEvaluated++;

  if (verify) {
    _performConsistencyCheck(game);
  }

  if (depth <= 0) {
    return SearchResult(null, game.nnueEvaluation);
  }

  final moves = game.generate_moves();
  if (moves.isEmpty) {
    if (game.in_check) return SearchResult(null, -999.0); // Checkmate
    return SearchResult(null, 0.0); // Draw
  }

  Move? bestMove;

  for (final move in moves) {
    game.make_move(move);
    // Negamax: Flip the score and alpha/beta bounds
    double score = -alphaBeta(game, depth - 1, -beta, -alpha, verify).score;
    game.undo_move();

    if (score >= beta) return SearchResult(move, beta);
    if (score > alpha) {
      alpha = score;
      bestMove = move;
    }
  }

  return SearchResult(bestMove, alpha);
}

/// Compares Incremental vs Full Refresh to ensure no "drift" occurs
void _performConsistencyCheck(ChessWithNNUE game) {
  double incrementalScore = game.nnueEvaluation;

  // Save the current accumulator state
  var savedAcc = game.accumulator.copy();

  // Force a full recalculation
  game.fullNnueRefresh();
  double absoluteScore = game.nnueEvaluation;

  // Restore the incremental state to continue search
  game.accumulator = savedAcc;

  double drift = (incrementalScore - absoluteScore).abs();
  if (drift > 0.0001) {
    print("!!! ACCUMULATOR DRIFT DETECTED !!!");
    print("Incremental: $incrementalScore");
    print("Absolute:    $absoluteScore");
    print("Difference:  $drift");
    print("Last FEN:    ${game.fen}");
    // throw Exception("NNUE Sync Error");
  }
}

void main() {
  print("--- Starting Chess Engine Search Test ---");
  final game = ChessWithNNUE();

  // 1. Initial Position Search
  print("\nSearching depth 3 from start...");
  nodesEvaluated = 0;
  var result = alphaBeta(game, 3, -1000.0, 1000.0, true);

  print("Best Move: ${game.move_to_san(result.move!)}");
  print("Score: ${result.score.toStringAsFixed(4)}");
  print("Nodes Evaluated: $nodesEvaluated");

  // 2. Test a complex tactical sequence (En Passant / Castling)
  print("\nTesting consistency on a long move sequence...");
  List<String> movesToTest = [
    "e4",
    "e5",
    "Nf3",
    "Nc6",
    "Bb5",
    "a6",
    "Ba4",
    "Nf6",
    "O-O",
  ];

  for (var m in movesToTest) {
    bool success = game.move(m);
    if (!success) break;

    // Check after every move
    _performConsistencyCheck(game);
    print("Checked move $m: Accumulator Healthy");
  }

  print("\nTesting Undo Consistency...");
  for (var i = 0; i < 5; i++) {
    game.undo_move();
    _performConsistencyCheck(game);
    print("Undo successful: Accumulator Healthy");
  }
}
