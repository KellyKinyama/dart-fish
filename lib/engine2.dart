import 'dart:math';
import "chess2.dart";
import 'consts.dart';

List minimax(String position, int depth, bool maxPlayer) {
  if (depth == 0 || Chess.fromFEN(position).in_checkmate) {
    return [Engine.fromFEN(position).eval(), null];
  }

  if (maxPlayer) {
    double maxEval = -inf;
    String? bestMove;
    for (var move in Chess.fromFEN(position).moves()) {
      Chess board = Chess.fromFEN(position);
      board.move(move);
      double eval = minimax(board.fen, depth - 1, false)[0];
      maxEval = max(maxEval, eval);
      if (maxEval == eval) {
        bestMove = move;
      }
    }
    return [maxEval, bestMove];
  } else {
    double minEval = inf;
    String? bestMove;
    for (var move in Chess.fromFEN(position).moves()) {
      Chess board = Chess.fromFEN(position);
      board.move(move);
      double eval = minimax(board.fen, depth - 1, true)[0];
      minEval = min(minEval, eval);
      if (minEval == eval) {
        bestMove = move;
      }
    }
    return [minEval, bestMove];
  }
}

class Engine extends Chess {
  Engine() : super();
  Engine.fromFEN(super.fen) : super.fromFEN();

  double eval() {
    if (insufficient_material) {
      return 0;
    }

    if (in_stalemate || in_threefold_repetition) {
      return -500;
    }

    if (in_checkmate) {
      return 9999999;
    }

    // Basic Material Evaluation
    double score = 0;
    for (int i = 0; i < board.length; i++) {
      final piece = board[i];
      if (piece is Piece) {
        final material = material_values[piece.type.toString()] ?? 0;
        final pieceTurn = turns[piece.color.toString()] ?? 1;
        final sideTurn = turns[turn.toString()] ?? 1;

        score += material * pieceTurn * sideTurn;
      }
    }
    for (String a in rows.keys.toList()) {
      for (int i = 1; i < 9; i++) {
        String square = a + i.toString();
        final piece = get(square);
        if (piece is Piece) {
          if (turns[turn.toString()] == 1) {
            //White's turn
            final table = PieceSquareTables[piece.type.toString()];
            final index = squarenum(square);
            score +=
                (table != null ? (table[index] ?? 0.0) : 0.0) *
                (turns[piece.color.toString()] ?? 1);
          } else {
            // Black's turn
            final table = PieceSquareTables[piece.type.toString()];
            final index = 63 - squarenum(square);
            score +=
                (table != null ? (table[index] ?? 0.0) : 0.0) *
                (turns[piece.color.toString()] ?? 1) *
                -1;
          }
        }
      }
    }

    return score;
  }

  String play() {
    List best = minimax(fen, 2, true);
    final String? bestMove = best[1] as String?;
    if (bestMove == null) {
      return "";
    }
    move(bestMove);
    var lastMove = getHistory({"verbose": true}).last;
    String lanstr = lastMove['from'] + lastMove['to'];
    if (lastMove['flags'].contains('p')) {
      String san = lastMove['san'].toString().toLowerCase();
      lanstr += san.substring(san.length - 1);
    }
    return lanstr;
  }

  bool moveLAN(String lanstr) {
    final e1 = get('e1');
    if (lanstr == "e1g1" && e1 is Piece && e1.type.name == 'k') {
      return move("O-O");
    }

    if (lanstr == "e1c1" && e1 is Piece && e1.type.name == 'k') {
      return move("O-O-O");
    }

    var coords = {'from': lanstr.substring(0, 2), 'to': lanstr.substring(2, 4)};
    if (lanstr.length > 4) {
      coords['promotion'] = lanstr[4].toLowerCase();
    }
    return move(coords);
  }

  int squarenum(String square) {
    int multiplier = int.parse(square.substring(square.length - 1)) - 1;
    final file = square.substring(0, 1);
    return 8 * multiplier + (rows[file] ?? 0);
  }
}
