// lib/nnue_persistence_ref.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'nnue_reference.dart'; // <-- NNUERef core

/// JSON serializer for NNUERef.
/// Saves/loads the float reference model (not quantized).
///
/// File schema (JSON):
/// {
///   "version": 1,
///   "arch": { "num_features": 40960, "M": 256, "K": 32, "scale_cp": 400.0 },
///   "l0": { "weights": [ [col0...], [col1...], ... ], "bias": [ ... ] },
///   "l1": { "weights": [ [col0...], [col1...], ... ], "bias": [ ... ] },
///   "l2": { "weights": [ [col0...], [col1...], ... ], "bias": [ ... ] }
/// }
class NNUESerializer {
  static const int _version = 1;

  static Map<String, dynamic> _toJson(NNUERef nnue) {
    Map<String, dynamic> packLayer(_LinearLike L) => {
      "weights": L.weights.map((col) => col.toList()).toList(),
      "bias": L.bias.toList(),
    };

    return {
      "version": _version,
      "arch": {
        "num_features": NNUERef.NUM_FEATURES,
        "M": NNUERef.M,
        "K": NNUERef.K,
        "scale_cp": NNUERef.SCALE_CP,
      },
      "l0": packLayer(_LinearLike(nnue.l0)),
      "l1": packLayer(_LinearLike(nnue.l1)),
      "l2": packLayer(_LinearLike(nnue.l2)),
    };
  }

  static void _fromJson(NNUERef nnue, Map<String, dynamic> root) {
    String _must(Map<String, dynamic> m, String k) {
      if (!m.containsKey(k)) throw FormatException("Missing key '$k'.");
      return k;
    }

    // version info (non-fatal if mismatched)
    final ver = root["version"];
    if (ver is! num || ver.toInt() != _version) {
      stdout.writeln(
        "⚠️  Serializer version differs (file=$ver, code=$_version). Attempting to load.",
      );
    }

    void unpackLayer(Map<String, dynamic> json, _LinearLike target) {
      final weightsDyn = json[_must(json, "weights")];
      final biasDyn = json[_must(json, "bias")];

      if (weightsDyn is! List || biasDyn is! List) {
        throw FormatException("Layer weights/bias must be lists.");
      }

      if (weightsDyn.length != target.inSize) {
        stdout.writeln(
          "⚠️  Column count mismatch: file=${weightsDyn.length} target=${target.inSize}",
        );
      }
      final minCols = weightsDyn.length < target.inSize
          ? weightsDyn.length
          : target.inSize;

      for (int c = 0; c < minCols; c++) {
        final colDyn = weightsDyn[c];
        if (colDyn is! List) {
          throw FormatException("Layer weight column $c must be a list.");
        }
        final col = target.weights[c];
        if (colDyn.length != col.length) {
          stdout.writeln(
            "⚠️  Column[$c] length mismatch: file=${colDyn.length} target=${col.length}",
          );
        }
        final minRows = colDyn.length < col.length ? colDyn.length : col.length;
        for (int r = 0; r < minRows; r++) {
          col[r] = (colDyn[r] as num).toDouble();
        }
      }

      if (biasDyn.length != target.outSize) {
        stdout.writeln(
          "⚠️  Bias length mismatch: file=${biasDyn.length} target=${target.outSize}",
        );
      }
      final minBias = biasDyn.length < target.outSize
          ? biasDyn.length
          : target.outSize;
      for (int i = 0; i < minBias; i++) {
        target.bias[i] = (biasDyn[i] as num).toDouble();
      }
    }

    unpackLayer(root[_must(root, "l0")], _LinearLike(nnue.l0));
    unpackLayer(root[_must(root, "l1")], _LinearLike(nnue.l1));
    unpackLayer(root[_must(root, "l2")], _LinearLike(nnue.l2));
  }

  /// Save returns void; throws on error.
  static Future<void> save(NNUERef nnue, String filePath) async {
    try {
      stdout.writeln('Encoding weights to JSON... (This may take a moment)');
      final jsonRoot = _toJson(nnue);
      final jsonString = const JsonEncoder.withIndent('  ').convert(jsonRoot);
      final file = File(filePath);
      await file.writeAsString(jsonString);
      stdout.writeln('✅ NNUE Weights successfully saved to: $filePath');
    } catch (e) {
      stderr.writeln('❌ Error saving weights: $e');
      rethrow;
    }
  }

  /// Load returns true on success, false if file missing or load failed.
  static Future<bool> load(NNUERef nnue, String filePath) async {
    final file = File(filePath);
    if (!await file.exists()) {
      stdout.writeln(
        '⚠️  No weights file found at $filePath. Starting with random weights.',
      );
      return false;
    }

    try {
      stdout.writeln('Reading weights from disk...');
      final jsonString = await file.readAsString();
      final dynamic decoded = jsonDecode(jsonString);
      if (decoded is! Map<String, dynamic>) {
        throw FormatException("Top-level JSON must be an object.");
      }
      _fromJson(nnue, decoded);
      stdout.writeln('✅ NNUE Weights successfully loaded from: $filePath');
      return true;
    } catch (e) {
      stderr.writeln('❌ Error loading weights: $e');
      return false;
    }
  }
}

class _LinearLike {
  final List<Float64List> weights;
  final Float64List bias;
  final int inSize;
  final int outSize;

  _LinearLike(dynamic layer)
    : weights = (layer.weights as List<Float64List>),
      bias = (layer.bias as Float64List),
      inSize = (layer as dynamic).inSize as int,
      outSize = (layer as dynamic).outSize as int;
}
