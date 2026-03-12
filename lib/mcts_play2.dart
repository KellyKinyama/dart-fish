import 'mcts3.dart'; // Ensure this contains your Engine and MCTS classes
import 'consts.dart';
// import 'engine3.dart';

void main() {
  // 1. Initialize the Engine with the starting position
  // The fromFEN constructor initializes our Zobrist keys.
  final engine = Engine.fromFEN(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  );

  print("=== MCTS GRAPH ENGINE DEBUG ===");
  print("Initial Position: ${engine.fen}");
  print("Initial Zobrist:  ${engine.zobristKey}");

  // 2. Perform a search
  // Using 2000 iterations to give the MCTS enough time to populate the graph
  print("\nMCTS is thinking (2000 iterations)...");

  final sw = Stopwatch()..start();

  // The play() method now uses MCTS search logic
  String bestMove = engine.play();

  sw.stop();

  print("---------------------------------");
  print("Best Move Found:  $bestMove");
  print("Time Taken:       ${sw.elapsedMilliseconds}ms");
  print("Nodes in Graph:   ${engine.lastSearchNodeCount ?? 'N/A'}");
  print("New Position:     ${engine.fen}");
  print("New Zobrist:      ${engine.zobristKey}");

  // 3. Verify the "Graph" behavior with a TRUE transposition
  // Use move orders that produce identical FEN (no EP squares)

  print("\n--- Testing Transposition (Graph) Detection ---");

  const startFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  final testEngine = Engine.fromFEN(startFEN);

  // Sequence A
  // 1 Nf3 Nf6 2 Nc3 Nc6
  testEngine.moveLAN("g1f3");
  testEngine.moveLAN("g8f6");
  testEngine.moveLAN("b1c3");
  testEngine.moveLAN("b8c6");

  int hashA = testEngine.zobristKey;
  String fenA = testEngine.fen;

  // Reset engine
  testEngine.load(startFEN);

  // Sequence B
  // 1 Nc3 Nc6 2 Nf3 Nf6
  testEngine.moveLAN("b1c3");
  testEngine.moveLAN("b8c6");
  testEngine.moveLAN("g1f3");
  testEngine.moveLAN("g8f6");

  int hashB = testEngine.zobristKey;
  String fenB = testEngine.fen;

  print("FEN A: $fenA");
  print("FEN B: $fenB");

  print("Hash A: $hashA");
  print("Hash B: $hashB");

  if (hashA == hashB && fenA == fenB) {
    print(
      "SUCCESS: True transposition detected! The MCTS graph can reuse this node.",
    );
  } else {
    print(
      "FAILED: Positions differ. Check incremental hash or move application.",
    );
  }
}
