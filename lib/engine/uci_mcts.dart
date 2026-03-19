import "dart:io";
// import "mcts_nnue.dart";
import "package:dart_cuda/aft_transformer_decoder.dart";
import "package:dart_cuda/gpu_tensor.dart";
import "package:dart_cuda/network_utils.dart";

import "chess_engine2.dart";

import 'chess_with_nnue.dart';
import "nnue_persistence_ref.dart";

Future<void> main() async {
  const modelPath = "nnue_static_model.json";
  const String weightPath = "chess_gpt.bin";

  // NNUEEngine engine = NNUEEngine();

  var engine = ChessWithNNUE();

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
      engine = ChessWithNNUE();
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
        engine = ChessWithNNUE();

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
        engine = ChessWithNNUE.fromFEN(fen);

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
      int depth = 6;
      final parts = cmd.split(" ");
      if (parts.length >= 3 && parts[1] == "depth") {
        depth = int.tryParse(parts[2]) ?? 6;
      }

      // Load Transformer policy head
      const vocabSize = 4098;
      const bigSize = 16;
      const blockSize = 16;

      final policyNet = TransformerDecoder(
        vocabSize: vocabSize,
        embedSize: bigSize,
        encoderEmbedSize: bigSize,
        numLayers: 2,
        numHeads: 4,
        blockSize: blockSize,
      );

      final dummyEnc = Tensor.zeros([1, bigSize]);

      await loadModuleBinary(policyNet, weightPath);
      final mcts = MCTS(
        engine,
        policyNet,
        dummyEnc,
        blockSize,
        iterations: 4000,
        cPuct: 1.2,
      );

      // Best move as LAN
      final bestLan = mcts.search();
      if (bestLan != null) {
        print("bestmove $bestLan");
        // engine.moveLAN(bestLan); // implement or reuse your own LAN applier
      } else {
        print('No legal moves found (game over).');
      }
      // final bestMoveLAN = engine.play(depth: depth);
      // print("bestmove $bestLan");
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
