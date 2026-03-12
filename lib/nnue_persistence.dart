import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'nnue_logic_batch.dart'; // Import your NNUE and LinearLayer classes

class NNUESerializer {
  /// Maps the NNUE layers into a flat list of Float64Lists for processing.
  static List<Float64List> _getParams(NNUE nnue) {
    List<Float64List> params = [];
    
    // Layer 0: Features to Accumulator
    for (var w in nnue.l0.weights) {
      params.add(w);
    }
    params.add(nnue.l0.bias);

    // Layer 1: Hidden Layer
    for (var w in nnue.l1.weights) {
      params.add(w);
    }
    params.add(nnue.l1.bias);

    // Layer 2: Output Layer
    for (var w in nnue.l2.weights) {
      params.add(w);
    }
    params.add(nnue.l2.bias);

    return params;
  }

  /// Saves the NNUE weights to a JSON file.
  /// Note: File will be large (~150MB+) due to the 41k feature set.
  static Future<void> save(NNUE nnue, String filePath) async {
    print('Encoding weights to JSON... (This may take a moment)');
    final List<Float64List> weightData = _getParams(nnue);
    
    // Convert Float64Lists to standard Lists for JSON encoding
    final List<List<double>> exportable = weightData.map((e) => e.toList()).toList();
    final String jsonString = jsonEncode(exportable);
    
    final File file = File(filePath);

    try {
      await file.writeAsString(jsonString);
      print('✅ NNUE Weights successfully saved to: $filePath');
    } catch (e) {
      print('❌ Error saving weights: $e');
    }
  }

  /// Loads weights from a JSON file back into the NNUE instance.
  static Future<void> load(NNUE nnue, String filePath) async {
    final File file = File(filePath);
    if (!await file.exists()) {
      print('⚠️ No weights file found at $filePath. Starting with random weights.');
      return;
    }

    try {
      print('Reading weights from disk...');
      final String jsonString = await file.readAsString();
      final List<dynamic> loadedData = jsonDecode(jsonString);
      final List<Float64List> targetParams = _getParams(nnue);

      if (loadedData.length != targetParams.length) {
        print('⚠️ Warning: Parameter count mismatch. Data might be corrupted.');
      }

      for (int i = 0; i < targetParams.length && i < loadedData.length; i++) {
        List<dynamic> sourceRow = loadedData[i];
        Float64List targetRow = targetParams[i];
        
        for (int j = 0; j < sourceRow.length && j < targetRow.length; j++) {
          targetRow[j] = (sourceRow[j] as num).toDouble();
        }
      }
      print('✅ NNUE Weights successfully loaded from: $filePath');
    } catch (e) {
      print('❌ Error loading weights: $e');
    }
  }
}