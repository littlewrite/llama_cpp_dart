import 'dart:async';
import 'package:typed_isolate/typed_isolate.dart';
import '../llama_cpp_dart.dart';

/// Child isolate that handles Llama model operations
class LlamaChild extends IsolateChild<LlamaResponse, LlamaCommand> {
  LlamaChild() : super(id: 1);

  bool shouldStop = false;
  Llama? llama;

  @override
  void run() {
    // This method is required by typed_isolate 3.0.0
    // The isolate is now ready to receive commands
  }

  @override
  void onData(LlamaCommand data) {
    if (data is LlamaStop) {
      _handleStop();
    } else if (data is LlamaClear) {
      _handleClear();
    } else if (data is LlamaLoad) {
      _handleLoad(data.path, data.modelParams, data.contextParams, 
          data.samplingParams, data.verbose, data.mmprojPath);
    } else if (data is LlamaPrompt) {
      _handlePrompt(data.prompt, data.promptId, data.images, data.slotId);
    } else if (data is LlamaInit) {
      _handleInit(data.libraryPath);
    } else if (data is LlamaEmbedd) {
      _handleEmbedding(data.prompt);
    } else if (data is LlamaDispose) {
      _handleDispose();
    } else if (data is LlamaSaveState) {
      _handleSaveState(data.slotId);
    } else if (data is LlamaLoadState) {
      _handleLoadState(data.slotId, data.data);
    } else if (data is LlamaLoadSession) {
      _handleLoadSession(data.slotId, data.path);
    } else if (data is LlamaFreeSlot) {
      _handleFreeSlot(data.slotId);
    }
  }

  void _handleDispose() {
    shouldStop = true;
    if (llama != null) {
      llama!.dispose();
      llama = null;
    }
  }

  /// Handle stop command
  void _handleStop() {
    shouldStop = true;
    send(
        LlamaResponse.confirmation(llama?.status ?? LlamaStatus.ready));
  }

  /// Handle clear command
  void _handleClear() {
    shouldStop = true;
    if (llama != null) {
      try {
        llama!.clear();
        send(LlamaResponse.confirmation(LlamaStatus.ready));
      } catch (e) {
        send(LlamaResponse.error("Error clearing context: $e"));
      }
    } else {
      send(LlamaResponse.error("Cannot clear: model not initialized"));
    }
  }

  /// Handle load command
  void _handleLoad(
      String path,
      ModelParams modelParams,
      ContextParams contextParams,
      SamplerParams samplingParams,
      bool verbose,
      String? mmprojPath) {
    try {
      llama = Llama(
        path,
        modelParams: modelParams,
        contextParams: contextParams,
        samplerParams: samplingParams,
        verbose: verbose,
        mmprojPath: mmprojPath,
      );
      send(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      send(LlamaResponse.error("Error loading model: $e"));
    }
  }

  /// Handle init command
  void _handleInit(String? libraryPath) {
    try {
      Llama.libraryPath = libraryPath;
      final _ = Llama.lib;
      send(LlamaResponse.confirmation(LlamaStatus.uninitialized));
    } catch (e) {
      send(LlamaResponse.error("Failed to open library: $e"));
    }
  }

  /// Handle embedding command
  void _handleEmbedding(String prompt) {
    shouldStop = false;
    if (llama == null) {
      send(LlamaResponse.error("Model not initialized"));
      return;
    }
    try {
      final embeddings = llama!.getEmbeddings(prompt);
      send(LlamaResponse(
        text: "",
        isDone: true,
        embeddings: embeddings,
        status: LlamaStatus.ready,
      ));
    } catch (e) {
      send(LlamaResponse.error("Embedding error: $e"));
    }
  }

  /// Handle prompt command
  void _handlePrompt(
      String prompt, String promptId, List<LlamaImage>? images, String? slotId) {
    shouldStop = false;
    _sendPrompt(prompt, promptId, images, slotId);
  }

  void _handleSaveState(String slotId) {
    if (llama == null) return;
    try {
      llama!.setSlot(slotId);

      final data = llama!.saveState();

      send(LlamaResponse.stateData(data));
    } catch (e) {
      send(
          LlamaResponse.error("Failed to save state for $slotId: $e"));
    }
  }

  void _handleLoadState(String slotId, dynamic data) {
    if (llama == null) return;
    try {
      try {
        llama!.createSlot(slotId);
      } catch (_) {
        // Ignore if slot already exists
      }
      llama!.setSlot(slotId);

      llama!.loadState(data);

      send(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      send(
          LlamaResponse.error("Failed to load state for $slotId: $e"));
    }
  }

  void _handleLoadSession(String slotId, String path) {
    if (llama == null) return;
    try {
      try {
        llama!.createSlot(slotId);
      } catch (_) {
        // Ignore if slot already exists
      }
      llama!.setSlot(slotId);

      final success = llama!.loadSession(path);
      if (success) {
        send(LlamaResponse.confirmation(LlamaStatus.ready));
      } else {
        send(LlamaResponse.error("Session file not found or invalid"));
      }
    } catch (e) {
      send(LlamaResponse.error("Failed to load session $path: $e"));
    }
  }

  void _handleFreeSlot(String slotId) {
    if (llama == null) return;
    try {
      llama!.freeSlot(slotId);
      send(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      // ignore: avoid_print
      print("Warning freeing slot $slotId: $e");
      send(LlamaResponse.confirmation(LlamaStatus.ready));
    }
  }

  Future<void> _sendPrompt(String prompt, String promptId,
      List<LlamaImage>? images, String? slotId) async {
    if (llama == null) {
      send(LlamaResponse.error("Model not initialized", promptId));
      return;
    }

    try {
      if (slotId != null) {
        try {
          llama!.createSlot(slotId);
          llama!.setSlot(slotId);
        } catch (e) {
          send(
              LlamaResponse.error("Slot allocation failed: $e", promptId));
          return;
        }
      } else {
        llama!.setSlot("default");
      }

      send(LlamaResponse(
          text: "",
          isDone: false,
          status: LlamaStatus.generating,
          promptId: promptId));

      Stream<String> tokenStream;

      if (images != null && images.isNotEmpty) {
        tokenStream = llama!.generateWithMedia(prompt, inputs: images);
      } else {
        llama!.setPrompt(prompt);
        tokenStream = llama!.generateText();
      }

      await for (final token in tokenStream) {
        if (shouldStop) break;
        send(LlamaResponse(
            text: token,
            isDone: false,
            status: LlamaStatus.generating,
            promptId: promptId));
      }

      send(LlamaResponse(
          text: "",
          isDone: true,
          status: LlamaStatus.ready,
          promptId: promptId));
    } catch (e) {
      send(
          LlamaResponse.error("Generation error: ${e.toString()}", promptId));
    }
  }
}