import 'dart:math';
import "chess3.dart";
import 'consts.dart';

// 1. FIXED MINIMAX: Now uses a single Engine instance and make/undo
List minimax(Engine board, int depth, bool maxPlayer) {
  // Check for terminal states using the board instance directly
  if (depth == 0 || board.in_checkmate || board.in_draw) {
    return [board.eval(), null];
  }

  // Get raw Move objects for make_move/undo_move
  final moves = board.generate_moves();

  if (maxPlayer) {
    double maxEval = -inf;
    Move? bestMove;
    for (var move in moves) {
      board.make_move(move);
      double evaluation = minimax(board, depth - 1, false)[0];
      board.undo_move();

      if (evaluation > maxEval) {
        maxEval = evaluation;
        bestMove = move;
      }
    }
    return [maxEval, bestMove];
  } else {
    double minEval = inf;
    Move? bestMove;
    for (var move in moves) {
      board.make_move(move);
      double evaluation = minimax(board, depth - 1, true)[0];
      board.undo_move();

      if (evaluation < minEval) {
        minEval = evaluation;
        bestMove = move;
      }
    }
    return [minEval, bestMove];
  }
}

class Engine extends Chess {
  Engine() : super();
  Engine.fromFEN(super.fen) : super.fromFEN();

  // 2. YOUR ORIGINAL EVAL (Simplified loop for speed)
  double eval() {
    if (insufficient_material) return 0;
    if (in_stalemate || in_threefold_repetition) return -500;
    if (in_checkmate)
      return -9999999; // Return negative because it's usually bad for the person moving

    double score = 0;

    // We only loop through valid squares (0x88 board)
    for (int i = 0; i < 128; i++) {
      if ((i & 0x88) != 0) continue;

      final piece = board[i];
      if (piece is Piece) {
        final pType = piece.type.toString();
        final material = material_values[pType] ?? 0;

        // Side multiplier: White 1, Black -1
        final sideMultiplier = (piece.color == Chess.WHITE) ? 1 : -1;
        score += material * sideMultiplier;

        // Piece Square Tables
        final table = PieceSquareTables[pType];
        if (table != null) {
          int index = (piece.color == Chess.WHITE) ? _to63(i) : 63 - _to63(i);
          score += (table[index] ?? 0.0) * sideMultiplier;
        }
      }
    }

    // Adjust score based on whose turn it is
    return (turn == Chess.WHITE) ? score : -score;
  }

  // Helper to convert 0x88 index to 0-63
  int _to63(int i) => (i >> 4) * 8 + (i & 7);

  // 3. UPDATED PLAY: Calls minimax with the current Engine instance
  String play() {
    // We pass 'this' (the engine) to minimax so it doesn't create new boards
    List best = minimax(this, 2, true);
    final Move? bestMove = best[1] as Move?;

    if (bestMove == null) return "";

    // Convert the Move object to the LAN string you expect
    String lanstr = bestMove.fromAlgebraic + bestMove.toAlgebraic;
    if (bestMove.promotion != null) {
      lanstr += bestMove.promotion!.name.toLowerCase();
    }

    make_move(bestMove);
    return lanstr;
  }

  bool moveLAN(String lanstr) {
    // Manual castling check if your move() function needs it
    final e1 = get('e1');
    if (lanstr == "e1g1" && e1 is Piece && e1.type.name == 'k')
      return move("O-O");
    if (lanstr == "e1c1" && e1 is Piece && e1.type.name == 'k')
      return move("O-O-O");

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

// void main() {
//   // 1. Initialize the Engine
//   // This calls load() internally, which initializes our Zobrist keys.
//   final engine = Engine.fromFEN(
//     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
//   );

//   print("--- Chess Engine Debug Session ---");
//   print("Initial FEN: ${engine.fen}");
//   print("Initial Hash: ${engine.zobristKey}");

//   // 2. Run a Perft test to verify Move Generation and Zobrist Integrity
//   // If make_move or undo_move has a bug, this will throw an Exception.
//   print("\nRunning Perft depth 3...");
//   try {
//     final nodes = engine.perft(3);
//     print("Perft 3 Success: $nodes nodes found.");
//   } catch (e) {
//     print("CRITICAL ERROR in Logic: $e");
//     return;
//   }

//   // 3. Evaluate the current position
//   print("\nStatic Evaluation: ${engine.eval()}");

//   // 4. Let the engine find the best move
//   print("\nEngine is thinking (Depth 2)...");
//   final Stopwatch timer = Stopwatch()..start();

//   String bestMoveLAN = engine.play(); // This calls our optimized minimax

//   timer.stop();

//   print("Best Move: $bestMoveLAN");
//   print("Time taken: ${timer.elapsedMilliseconds}ms");
//   print("New FEN: ${engine.fen}");
//   print("New Hash: ${engine.zobristKey}");

//   // 5. Example of making a manual move using LAN
//   print("\nApplying opponent move 'e7e5'...");
//   bool success = engine.moveLAN("e7e5");
//   if (success) {
//     print("Success! New Turn: ${engine.turn}");
//   } else {
//     print("Invalid move provided.");
//   }
// }

// void main() {
//   final engine = Engine.fromFEN(
//     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
//   );

//   print("Initial Key: ${engine.zobristKey}");

//   // This will now run without RangeErrors!
//   try {
//     int nodes = engine.perft(3);
//     print("Perft 3: $nodes nodes. Zobrist is stable.");
//   } catch (e) {
//     print("Logic Error: $e");
//   }
// }

void main() {
  final engine = Engine();

  // Define critical test positions
  // 1. Starting Position
  // 2. "Kiwipete" - Famous for testing castling/EP/complex tactics
  // 3. Endgame with promotion and draw potential
  final Map<String, String> testPositions = {
    "Start Pos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Kiwipete":
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "Promotion/EP": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
  };

  print("=== ZOBRIST STATE INTEGRITY SUITE ===\n");

  testPositions.forEach((name, fen) {
    print("Testing Position: $name");
    engine.load(fen);

    // We store the hash calculated from scratch by load()
    final int startKey = engine.zobristKey;
    print("  Initial Hash: $startKey");

    try {
      // Depth 3 is usually enough to catch "Hash Drift"
      // where make/undo don't return to the original value.
      final nodes = engine.perft(3);

      // Verify that after thousands of moves and undos, we are back to start
      if (engine.zobristKey != startKey) {
        print("  FAILED: Hash Drift detected!");
        print("  Expected: $startKey");
        print("  Actual:   ${engine.zobristKey}");
      } else {
        print("  SUCCESS: $nodes nodes. Hash is rock solid.");
      }
    } catch (e) {
      print("  CRITICAL ERROR: $e");
    }
    print("-" * 40);
  });

  // --- ENGINE PLAY TEST ---
  print("\n=== ENGINE PERFORMANCE TEST ===");
  engine.load("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  final sw = Stopwatch()..start();
  // Depth 4 is a good benchmark for a simple minimax
  String bestMove = engine.play();
  sw.stop();

  print("Engine Move: $bestMove");
  print("Search Time: ${sw.elapsedMilliseconds}ms");

  // Final check: Does the FEN after play() reflect the hash?
  final int postPlayHash = engine.zobristKey;
  final int verificationHash = engine.generateZobristKey(); // Manual re-calc

  if (postPlayHash == verificationHash) {
    print("Post-move Hash Consistency: VALID");
  } else {
    print("Post-move Hash Consistency: INVALID (Incremental update failed)");
  }
}
