import "dart:io";
// import "nnue_engine.dart";
import "nnue_minimax.dart";
import "nnue_persistence_ref.dart";

Future<void> main() async {
  const modelPath = "nnue_static_model.json";

  NNUEEngine engine = NNUEEngine();

  // Load NNUE weights
  final loaded = await NNUESerializer.load(engine.nnue, modelPath);
  stderr.writeln(
    loaded
        ? "info string Loaded NNUE weights"
        : "info string Using random NNUE weights",
  );

  while (true) {
    final cmd = stdin.readLineSync();
    if (cmd == null) continue;

    // ----------------------
    // UCI handshake
    // ----------------------
    if (cmd == "uci") {
      print("id name Dartfish-NNUE");
      print("id author Kelly Kinyama");
      print("uciok");
      continue;
    }

    if (cmd == "isready") {
      print("readyok");
      continue;
    }

    if (cmd == "ucinewgame") {
      engine = NNUEEngine();
      continue;
    }

    // ----------------------
    // Position command
    // ----------------------
    if (cmd.startsWith("position")) {
      // Format:
      // position startpos moves e2e4 e7e5
      // position fen <FEN> moves ...
      final parts = cmd.split(" ");

      if (parts.length >= 2 && parts[1] == "startpos") {
        engine = NNUEEngine();

        // Apply moves if any
        final movesIndex = parts.indexOf("moves");
        if (movesIndex != -1) {
          for (int i = movesIndex + 1; i < parts.length; i++) {
            engine.moveLAN(parts[i]);
          }
        }
      } else if (parts.length >= 3 && parts[1] == "fen") {
        // Extract fen until "moves"
        final fenParts = <String>[];
        int i = 2;
        while (i < parts.length && parts[i] != "moves") {
          fenParts.add(parts[i]);
          i++;
        }
        final fen = fenParts.join(" ");
        engine = NNUEEngine.fromFEN(fen);

        // Moves after fen
        if (i < parts.length && parts[i] == "moves") {
          for (int j = i + 1; j < parts.length; j++) {
            engine.moveLAN(parts[j]);
          }
        }
      }

      continue;
    }

    // ----------------------
    // go command
    // ----------------------
    if (cmd.startsWith("go")) {
      // Optional: parse depth from "go depth X"
      int depth = 2;
      final parts = cmd.split(" ");
      if (parts.length >= 3 && parts[1] == "depth") {
        depth = int.tryParse(parts[2]) ?? 2;
      }

      final bestMoveLAN = engine.play(depth: depth);
      print("bestmove $bestMoveLAN");
      continue;
    }

    // ----------------------
    // quit
    // ----------------------
    if (cmd == "quit") {
      break;
    }
  }
}
