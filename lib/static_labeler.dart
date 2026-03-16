// lib/static_labeler.dart
import 'engine.dart'; // your Engine class that extends Chess from chess.dart

/// Compute a static evaluation in centipawns from a FEN.
/// This uses your handcrafted Engine.eval() and is independent of NNUE.
double staticEvalCpFromFen(String fen) {
  final engine = Engine.fromFEN(fen);
  return engine.eval();
}
