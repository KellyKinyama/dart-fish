import 'dart:typed_data';
import 'chess3.dart';
import 'constants.dart';
import 'nnue_logic_batch.dart'; // Assuming the NNUE code from before is here

class ChessWithNNUE extends Chess {
  late NNUE nnue;
  late NnueAccumulator accumulator;

  ChessWithNNUE() : super() {
    nnue = NNUE(); // Loads weights
    accumulator = NnueAccumulator(NNUE.M);
    fullNnueRefresh();
  }

  /// Maps 0x88 square (0-127) to 64-square (0-63)
  int _x88to64(int sq) => (sq >> 4) * 8 + (sq & 7);

  /// Completely re-syncs the NNUE accumulator with the current board state
  void fullNnueRefresh() {
    int whiteKing = _x88to64(kings[Color.WHITE]);
    int blackKing = _x88to64(kings[Color.BLACK]);

    List<int> whiteFeatures = [];
    List<int> blackFeatures = [];

    for (int i = 0; i < 128; i++) {
      if ((i & 0x88) != 0) continue; // Skip invalid 0x88 slots
      final p = board[i];
      if (p == null || p.type == PieceType.KING) continue;

      int sq64 = _x88to64(i);
      whiteFeatures.add(getHalfKPIndex(whiteKing, sq64, p, false));
      blackFeatures.add(getHalfKPIndex(blackKing, sq64, p, true));
    }

    accumulator.refresh(nnue.l0, whiteFeatures, 0); // White view
    accumulator.refresh(nnue.l0, blackFeatures, 1); // Black view
  }

  /// Generates the HalfKP index for a piece relative to the king.
  int getHalfKPIndex(int kingSq, int pieceSq, Piece piece, bool flip) {
    int sq = flip ? pieceSq ^ 56 : pieceSq;
    int kSq = flip ? kingSq ^ 56 : kingSq;
    int pIdx =
        piece.type.shift * 2 +
        (flip
            ? (piece.color == Color.WHITE ? 1 : 0)
            : (piece.color == Color.WHITE ? 0 : 1));
    // Simplified HalfKP mapping
    return sq + (pIdx + kSq * 10) * 64;
  }

  // int getHalfKPIndex(
  //   int kingSq,
  //   int pieceSq,
  //   PieceType type,
  //   Color color,
  //   bool flip,
  // ) {
  //   int sq = flip ? pieceSq ^ 56 : pieceSq;
  //   int kSq = flip ? kingSq ^ 56 : kingSq;

  //   // Normalize color: 0 for "side to move", 1 for "enemy"
  //   // If we are flipping (Black's perspective), White pieces (index 0) become "enemy" (1)
  //   int pColorIdx = color == Color.WHITE ? 0 : 1;
  //   int side = flip ? 1 - pColorIdx : pColorIdx;

  //   // Standard NNUE piece indexing: Pawn=0, Knight=1, ..., Queen=4
  //   // We use type.index if your enum is ordered P, N, B, R, Q, K
  //   int pIdx = type.shift * 2 + side;

  //   return sq + (pIdx + kSq * 10) * 64;
  // }

  // @override
  // void make_move(Move move) {
  //   // Check if it's a king move (needs full refresh)
  //   bool isKingMove = move.piece == PieceType.KING;

  //   super.make_move(move);

  //   if (isKingMove) {
  //     _fullNnueRefresh();
  //   } else {
  //     _updateNnueIncremental(move);
  //   }
  // }

  void _updateNnueIncremental(Move move) {
    final us = move.color;
    int wKing = _x88to64(kings[Color.WHITE]);
    int bKing = _x88to64(kings[Color.BLACK]);

    List<int> addedW = [];
    List<int> removedW = [];
    List<int> addedB = [];
    List<int> removedB = [];

    int from64 = _x88to64(move.from);
    int to64 = _x88to64(move.to);

    // 1. Piece moved
    removedW.add(getHalfKPIndex(wKing, from64, Piece(move.piece, us), false));
    removedB.add(getHalfKPIndex(bKing, from64, Piece(move.piece, us), true));

    if (move.promotion != null) {
      addedW.add(
        getHalfKPIndex(wKing, to64, Piece(move.promotion!, us), false),
      );
      addedB.add(getHalfKPIndex(bKing, to64, Piece(move.promotion!, us), true));
    } else {
      addedW.add(getHalfKPIndex(wKing, to64, Piece(move.piece, us), false));
      addedB.add(getHalfKPIndex(bKing, to64, Piece(move.piece, us), true));
    }

    // 2. Captures
    if (move.captured != null) {
      int capSq = to64;
      if ((move.flags & BITS_EP_CAPTURE) != 0) {
        capSq = _x88to64(us == Color.WHITE ? move.to + 16 : move.to - 16);
      }
      removedW.add(
        getHalfKPIndex(
          wKing,
          capSq,
          Piece(move.captured!, Chess.swap_color(us)),
          false,
        ),
      );
      removedB.add(
        getHalfKPIndex(
          bKing,
          capSq,
          Piece(move.captured!, Chess.swap_color(us)),
          true,
        ),
      );
    }

    // Apply to accumulator
    accumulator.update(nnue.l0, addedW, removedW, 0);
    accumulator.update(nnue.l0, addedB, removedB, 1);
  }

  double get nnueEvaluation => nnue.evaluate(accumulator, turn);

  @override
  void make_move(Move move) {
    // 1. Update the board state using the parent class
    super.make_move(move);

    // 2. Update the NNUE "Brain"
    if (move.piece == PieceType.KING) {
      fullNnueRefresh();
    } else {
      // Standard move: Add 'to', Remove 'from'
      _applyMoveToAccumulator(move, isUndo: false);
    }
  }

  @override
  Move? undo_move() {
    // Capture the move we are about to undo
    if (history.isEmpty) return null;
    final move = history.last.move;
    final wasKingMove = move.piece == PieceType.KING;

    // 1. Revert board state
    final undoneMove = super.undo_move();

    // 2. Revert NNUE
    if (wasKingMove) {
      fullNnueRefresh();
    } else {
      // Undo: Remove 'to', Add 'from'
      _applyMoveToAccumulator(move, isUndo: true);
    }
    return undoneMove;
  }

  /// Shared logic for updating the accumulator
  void _applyMoveToAccumulator(Move move, {required bool isUndo}) {
    final us = move.color;
    final them = (us == Color.WHITE) ? Color.BLACK : Color.WHITE;

    int wKing = _x88to64(kings[Color.WHITE]);
    int bKing = _x88to64(kings[Color.BLACK]);

    List<int> addedW = [];
    List<int> removedW = [];
    List<int> addedB = [];
    List<int> removedB = [];

    int from64 = _x88to64(move.from);
    int to64 = _x88to64(move.to);

    // Define what happened in the move
    // Note: We use the move's stored data because the board has already changed
    void stageChange(int sq64, PieceType type, Color color, bool isAddition) {
      int idxW = getHalfKPIndex(wKing, sq64, Piece(type, color), false);
      int idxB = getHalfKPIndex(bKing, sq64, Piece(type, color), true);

      if (isAddition) {
        addedW.add(idxW);
        addedB.add(idxB);
      } else {
        removedW.add(idxW);
        removedB.add(idxB);
      }
    }

    // Handle the moving piece
    stageChange(from64, move.piece, us, false); // Removed from start
    if (move.promotion != null) {
      stageChange(to64, move.promotion!, us, true); // Added as Queen/etc
    } else {
      stageChange(to64, move.piece, us, true); // Added at destination
    }

    // Handle captures
    if (move.captured != null) {
      int capSq = to64;
      if ((move.flags & BITS_EP_CAPTURE) != 0) {
        capSq = _x88to64(us == Color.WHITE ? move.to + 16 : move.to - 16);
      }
      stageChange(capSq, move.captured!, them, false); // Captured piece removed
    }

    // Handle Castling (Rook displacement)
    if ((move.flags & (BITS_KSIDE_CASTLE | BITS_QSIDE_CASTLE)) != 0) {
      int rFrom, rTo;
      if ((move.flags & BITS_KSIDE_CASTLE) != 0) {
        rFrom = move.to + 1;
        rTo = move.to - 1;
      } else {
        rFrom = move.to - 2;
        rTo = move.to + 1;
      }
      stageChange(_x88to64(rFrom), PieceType.ROOK, us, false);
      stageChange(_x88to64(rTo), PieceType.ROOK, us, true);
    }

    // If this is an UNDO, we flip the logic:
    // What was "added" is now being "removed" and vice-versa.
    if (isUndo) {
      accumulator.update(nnue.l0, removedW, addedW, 0);
      accumulator.update(nnue.l0, removedB, addedB, 1);
    } else {
      accumulator.update(nnue.l0, addedW, removedW, 0);
      accumulator.update(nnue.l0, addedB, removedB, 1);
    }
  }
}

void main() {
  print("Initializing ChessWithNNUE Engine...");

  // 1. Initialize the board with NNUE capabilities
  // This automatically runs _fullNnueRefresh() for the starting position
  final game = ChessWithNNUE();

  // 2. Print initial evaluation
  // Evaluation is from the perspective of the side whose turn it is
  print("Initial Position Eval: ${game.nnueEvaluation.toStringAsFixed(4)}");

  // 3. Make a move using SAN (Standard Algebraic Notation)
  // The 'move' method in the Chess library calls our overridden 'make_move'
  print("\nMoving: 1. e4");
  game.move("e4");
  print("Eval after 1. e4: ${game.nnueEvaluation.toStringAsFixed(4)}");

  // 4. Make a move for Black
  print("Moving: 1... e5");
  game.move("e5");
  print("Eval after 1... e5: ${game.nnueEvaluation.toStringAsFixed(4)}");

  // 5. Demonstrate Capture (Knight takes Pawn)
  // This triggers the incremental capture logic in _updateNnueIncremental
  game.load("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
  game.fullNnueRefresh(); // Refresh after a manual FEN load

  print("\nPosition: Knight can capture on e5");
  print("Before capture: ${game.nnueEvaluation.toStringAsFixed(4)}");

  game.move("Nxe5");
  print("After Nxe5: ${game.nnueEvaluation.toStringAsFixed(4)}");

  // 6. Accessing the underlying board if needed
  print("\nCurrent Turn: ${game.turn == Color.WHITE ? 'White' : 'Black'}");
}
