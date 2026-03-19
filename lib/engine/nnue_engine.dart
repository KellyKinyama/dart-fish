import 'dart:math';
import 'chess.dart';
import 'chess_with_nnue.dart';
import 'constants.dart';
import 'nnue_reference.dart';
import 'tt.dart';

const double INF = 1e12;
const int QS_MAX_DEPTH = 32; // prevents runaway quiescence recursion
const double FUTILITY_MARGIN = 300.0;

const int HISTORY_MAX = 1 << 20;

bool isQuiet(Move mv) =>
    (mv.flags & (BITS_CAPTURE | BITS_PROMOTION | BITS_EP_CAPTURE)) == 0;

/// ===========================================================
///   UNIVERSAL STATIC NNUE BOARD EVALUATOR
/// ===========================================================
extension NNUEBoardEval on NNUERef {
  double evaluateBoard(List<Piece?> board, Color turn) {
    final acc = newAccumulator();
    refreshAccumulator(acc, board); // full rebuild
    return evaluate(acc, turn);
  }
}

bool squareInBoard(int sq) => sq >= 0 && sq < 64;

int captureScore(ChessWithNNUE pos, Move mv) {
  if (!squareInBoard(mv.from) || !squareInBoard(mv.to)) return 0;
  final victim = pos.board[mv.to];
  final attacker = pos.board[mv.from];
  if (victim == null || attacker == null) return 0;
  return (victim.type.shift + 1) * 10 - attacker.type.shift;
}

// ================================================================
// KILLER + HISTORY TABLES
// ================================================================

class SearchTables {
  final List<List<Move?>> killers = List.generate(
    128,
    (_) => <Move?>[null, null],
  );
  final history = List.generate(128, (_) => List.filled(64, 0));

  void addKiller(int ply, Move mv) {
    if (!squareInBoard(mv.from) || !squareInBoard(mv.to)) return;
    if (killers[ply][0] == mv) return;
    killers[ply][1] = killers[ply][0];
    killers[ply][0] = mv;
  }

  void addHistory(Color side, Move mv, int depth) {
    if (!squareInBoard(mv.from) || !squareInBoard(mv.to)) return;
    final s = side == Color.WHITE ? 0 : 1;
    final idx = s * 64 + mv.from;
    history[idx][mv.to] += depth * depth;

    if (history[idx][mv.to] > HISTORY_MAX) {
      for (var row in history) {
        for (int i = 0; i < row.length; i++) {
          row[i] >>= 1;
        }
      }
    }
  }

  int historyScore(Color side, Move mv) {
    if (!squareInBoard(mv.from) || !squareInBoard(mv.to)) return 0;
    final s = side == Color.WHITE ? 0 : 1;
    return history[s * 64 + mv.from][mv.to];
  }
}

// ================================================================
// STATIC EVALUATION (NO QUIESCENCE)
// ================================================================

double staticEval(ChessWithNNUE pos, int color) {
  return color * pos.nnue.evaluateBoard(pos.board, pos.turn);
}

/// ===========================================================
///   ROOT MOVE CLASS  (score + move + helpers)
/// ===========================================================
class RootMove {
  final Move move;
  double score;

  RootMove(this.move, this.score);

  @override
  String toString() => "${move.fromAlgebraic}${move.toAlgebraic} ($score)";
}

/// ===========================================================
///   NEGAMAX + ALPHA-BETA PRUNING
/// ===========================================================
/// color = +1 (side to move), -1 (opponent)
double negamax(
  ChessWithNNUE pos,
  int depth,
  double alpha,
  double beta,
  int color,
  int ply,
  TranspositionTable tt,
  SearchTables tables,
) {
  final int hash = pos.zobristKey; // MUST use zobrist hash from your engine

  // --- Probe Transposition Table ---
  // --- Probe Transposition Table ---
  final HashEntry? entry = tt.probe(hash);

  const bool debugTT = false; // set true to enable TT debug logs

  if (debugTT) {
    if (entry == null) {
      print("TT MISS  hash=$hash");
    } else {
      print(
        "TT HIT   hash=$hash  "
        "storedDepth=${entry.depth}  searchDepth=$depth  "
        "bound=${entry.bound}  score=${entry.score}",
      );
    }
  }

  if (entry != null && entry.depth >= depth) {
    switch (entry.bound) {
      case Bound.EXACT:
        if (debugTT) print("TT EXACT → return ${entry.score}");
        return entry.score;

      case Bound.LOWER:
        if (debugTT) print("TT LOWER → alpha=$alpha → ${entry.score}");
        alpha = max(alpha, entry.score);
        break;

      case Bound.UPPER:
        if (debugTT) print("TT UPPER → beta=$beta → ${entry.score}");
        beta = min(beta, entry.score);
        break;
    }

    if (alpha >= beta) {
      if (debugTT) {
        print("TT CUTOFF  alpha=$alpha beta=$beta  → return ${entry.score}");
      }
      return entry.score;
    }
  }

  // Terminal
  if (depth == 0) {
    return qsearch(pos, alpha, beta, color);
  }

  // Leaf or terminal
  // if (depth == 0 || pos.in_checkmate) {
  //   final eval = color * pos.nnue.evaluateBoard(pos.board, pos.turn);

  //   tt.store(
  //     HashEntry(hash: hash, score: eval, depth: depth, bound: Bound.EXACT),
  //   );
  //   return eval;
  // }

  double best = -INF;
  Move? bestMove;

  final moves = pos.generate_moves();
  // MOVE ORDERING
  Move? hashMove = entry?.move;

  moves.sort((a, b) {
    // if (!squareInBoard(a.from) || !squareInBoard(a.to)) return 1;
    // if (!squareInBoard(b.from) || !squareInBoard(b.to)) return -1;

    if (a == hashMove) return -1;
    if (b == hashMove) return 1;

    final ac = !isQuiet(a);
    final bc = !isQuiet(b);

    if (ac && bc) return captureScore(pos, b) - captureScore(pos, a);
    if (ac) return -1;
    if (bc) return 1;

    int as = tables.historyScore(pos.turn, a);
    int bs = tables.historyScore(pos.turn, b);

    if (tables.killers[ply].contains(a)) as += 50000;
    if (tables.killers[ply].contains(b)) bs += 50000;

    return bs - as;
  });

  final us = pos.turn;

  int movesFound = 0;
  int pvMovesFound = 0;

  for (final mv in moves) {
    pos.make_move(mv);

    // FUTILITY
    if (depth == 1 && isQuiet(mv) && !pos.in_check) {
      double eval = -staticEval(pos, color);
      if (eval + FUTILITY_MARGIN <= alpha) {
        pos.undo_move();
        movesFound++;
        continue;
      }
    }

    int newDepth = depth - 1;

    // LMR
    // if (depth >= 3 && movesFound > 0 && isQuiet(mv) && !pos.in_check) {
    //   newDepth = max(1, newDepth - 1);
    // }

    if (!pos.king_attacked(us)) {
      movesFound++;

      double score;
      if (pvMovesFound > 0) {
        if (depth >= 3 && movesFound > 0 && isQuiet(mv) && !pos.in_check) {
          score = -negamax(
            pos,
            newDepth - 1,
            -alpha - 1,
            -alpha,
            -color,
            ply + 1,
            tt,
            tables,
          );
        } else {
          score = alpha + 1;
        }
        if (score > alpha) {
          score = -negamax(
            pos,
            depth - 1,
            -alpha - 1,
            -alpha,
            -color,
            ply + 1,
            tt,
            tables,
          );
        }

        if (score > alpha && score < beta) {
          score = -negamax(
            pos,
            depth - 1,
            -beta,
            -alpha,
            -color,
            ply + 1,
            tt,
            tables,
          );
        }
      } else {
        if (depth >= 3 && movesFound > 0 && isQuiet(mv) && !pos.in_check) {
          score = -negamax(
            pos,
            newDepth - 1,
            -beta,
            -alpha,
            -color,
            ply + 1,
            tt,
            tables,
          );
        } else {
          score = alpha + 1;
        }
        if (score > alpha) {
          score = -negamax(
            pos,
            depth - 1,
            -beta,
            -alpha,
            -color,
            ply + 1,
            tt,
            tables,
          );
        }
      }

      pos.undo_move();

      if (score > best) {
        best = score;
        bestMove = mv;
      }

      if (score > alpha) {
        alpha = score;
        pvMovesFound++;

        if (isQuiet(mv)) {
          tables.addHistory(pos.turn, mv, depth);
          tables.addKiller(ply, mv);
        }
      }

      if (alpha >= beta) break;
    } else {
      print("illegal move");
      pos.undo_move();
    }
  }

  if (movesFound == 0) {
    // print("Checkmated");
    if (pos.king_attacked(us == Color.WHITE ? Color.BLACK : Color.WHITE)) {
      Bound bound;
      if (best <= alpha) {
        bound = Bound.UPPER;
      } else if (best >= beta) {
        bound = Bound.LOWER;
      } else {
        bound = Bound.EXACT;
      }
      tt.store(
        HashEntry(
          hash: hash,
          score: -(9999 - ply).toDouble(),
          depth: depth,
          move: bestMove,
          bound: bound,
        ),
      );
      return -(9999 - ply).toDouble();
    } else {
      Bound bound;
      if (best <= alpha) {
        bound = Bound.UPPER;
      } else if (best >= beta) {
        bound = Bound.LOWER;
      } else {
        bound = Bound.EXACT;
      }
      print("Checkmated");
      tt.store(
        HashEntry(
          hash: hash,
          score: 0.toDouble(),
          depth: depth,
          move: bestMove,
          bound: bound,
        ),
      );
      return -(9999 - ply).toDouble();
    }
  }

  // --- Store into TT ---
  Bound bound;
  if (best <= alpha) {
    bound = Bound.UPPER;
  } else if (best >= beta) {
    bound = Bound.LOWER;
  } else {
    bound = Bound.EXACT;
  }

  tt.store(
    HashEntry(
      hash: hash,
      score: best,
      depth: depth,
      move: bestMove,
      bound: bound,
    ),
  );

  return best;
}

// ================================================================
// QUIESCENCE SEARCH — CAPTURES ONLY, SAFE VERSION
// ================================================================

double qsearch(
  ChessWithNNUE pos,
  double alpha,
  double beta,
  int color, {
  int qdepth = 0,
}) {
  if (qdepth >= QS_MAX_DEPTH) return alpha;

  double stand = color * pos.nnue.evaluateBoard(pos.board, pos.turn);

  if (stand >= beta) return stand;
  if (stand > alpha) alpha = stand;

  final moves = pos.generate_moves().where((m) => !isQuiet(m)); // captures only

  for (final mv in moves) {
    pos.make_move(mv);

    double score = -qsearch(pos, -beta, -alpha, -color, qdepth: qdepth + 1);

    pos.undo_move();

    if (score >= beta) return score;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

/// ===========================================================
///   ROOT NEGAMAX SEARCH (one depth)
/// ===========================================================
RootMove rootSearch(
  ChessWithNNUE pos,
  int depth,
  TranspositionTable tt,
  SearchTables tables,
) {
  final moves = pos.generate_moves({'legal': true});
  if (moves.isEmpty) {
    return RootMove(null as Move, pos.nnue.evaluateBoard(pos.board, pos.turn));
  }

  double alpha = -INF;
  double beta = INF;

  RootMove? bestRM;

  for (final mv in moves) {
    pos.make_move(mv);

    double score = -negamax(pos, depth - 1, -beta, -alpha, -1, 1, tt, tables);

    pos.undo_move();

    if (score > alpha) {
      alpha = score;
      bestRM = RootMove(mv, score);
    }
  }

  return bestRM!;
}

/// ===========================================================
///   ITERATIVE DEEPENING DRIVER
/// ===========================================================
RootMove iterativeDeepening(ChessWithNNUE pos, int maxDepth) {
  final tt = TranspositionTable();
  final tables = SearchTables();

  RootMove? best;

  for (int depth = 1; depth <= maxDepth; depth++) {
    best = rootSearch(pos, depth, tt, tables);
    print(
      "info depth $depth selmove ${best.move.fromAlgebraic}${best.move.toAlgebraic} score ${best.score}",
    );
  }

  return best!;
}

/// ===========================================================
///   MAIN ENGINE WRAPPER
/// ===========================================================
class NNUEEngine extends ChessWithNNUE {
  NNUEEngine() : super();
  NNUEEngine.fromFEN(String fen) : super() {
    load(fen);
  }

  double eval() => nnue.evaluateBoard(board, turn);

  /// Choose and play the best move using iterative deepening + negamax + alpha-beta
  String play({int depth = 4}) {
    final best = iterativeDeepening(this, depth);
    final mv = best.move;

    make_move(mv);

    final last = history.last.move;
    String lan = last.fromAlgebraic + last.toAlgebraic;

    if (last.promotion != null) {
      lan += last.promotion!.name[0].toLowerCase();
    }
    return lan;
  }

  /// LAN interface
  bool moveLAN(String lan) {
    if (lan.length < 4) return false;

    final from = lan.substring(0, 2);
    final to = lan.substring(2, 4);

    PieceType? promo;
    if (lan.length > 4) promo = _promo(lan[4]);

    final moves = generate_moves();
    for (final m in moves) {
      if (m.fromAlgebraic == from && m.toAlgebraic == to) {
        if ((m.flags & BITS_PROMOTION) != 0) {
          if (promo == null || promo != m.promotion) continue;
        } else {
          if (promo != null) continue;
        }
        make_move(m);
        return true;
      }
    }

    return false;
  }

  PieceType? _promo(String ch) {
    switch (ch.toLowerCase()) {
      case 'q':
        return PieceType.QUEEN;
      case 'r':
        return PieceType.ROOK;
      case 'b':
        return PieceType.BISHOP;
      case 'n':
        return PieceType.KNIGHT;
      default:
        return null;
    }
  }
}
