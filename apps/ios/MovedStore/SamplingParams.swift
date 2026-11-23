// Sampling parameters for the app's replies: the model folder's generation_config.json, overridden by Settings. The draw
// itself is the engine's C sampler (DmengModel.sample: penalties, temperature, top-k, top-p, min-p, one categorical draw
// keyed by the seed and the token index), the same on the Mac and the phone.
import Foundation

public struct SamplingParams: Sendable, Equatable, Codable {
    public var temperature: Float
    public var topK: Int  // 0: off
    public var topP: Float  // 1: off
    public var minP: Float  // 0: off (relative to the most likely token after top-k / temperature)
    public var presencePenalty: Float  // 0: off
    public var repetitionPenalty: Float  // 1: off
    public var frequencyPenalty: Float  // 0: off
    public var seed: UInt64

    public init(temperature: Float, topK: Int = 0, topP: Float = 1, minP: Float = 0, presencePenalty: Float = 0,
                repetitionPenalty: Float = 1, frequencyPenalty: Float = 0, seed: UInt64 = 0) {
        self.temperature = temperature; self.topK = topK; self.topP = topP; self.minP = minP
        self.presencePenalty = presencePenalty; self.repetitionPenalty = repetitionPenalty
        self.frequencyPenalty = frequencyPenalty; self.seed = seed
    }

    /// Argmax.
    public static let greedy = SamplingParams(temperature: 0)
    /// The release's generation_config.json (Qwen3.6-35B-A3B): do_sample, temperature 1.0, top_k 20, top_p 0.95.
    public static let modelDefault = SamplingParams(temperature: 1.0, topK: 20, topP: 0.95)
    /// The Qwen3.6-35B-A3B model card's presets ("We recommend using the following set of sampling parameters"):
    /// thinking mode, general tasks: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repetition 1.0.
    public static let qwenThinking = SamplingParams(temperature: 1.0, topK: 20, topP: 0.95, presencePenalty: 1.5)
    /// thinking mode, precise coding tasks: temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.
    public static let qwenThinkingCoding = SamplingParams(temperature: 0.6, topK: 20, topP: 0.95)
    /// instruct (non-thinking) mode: temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5.
    public static let qwenNonThinking = SamplingParams(temperature: 0.7, topK: 20, topP: 0.8, presencePenalty: 1.5)

    /// No randomness: the argmax of the (penalised) logits.
    public var isGreedy: Bool { temperature <= 0 }
    public var hasPenalties: Bool { presencePenalty != 0 || repetitionPenalty != 1 || frequencyPenalty != 0 }
    /// The engine's plain argmax path applies (no Sampler is built).
    public var isPlainArgmax: Bool { isGreedy && !hasPenalties }

    /// generation_config.json in `dir` (temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty), each key
    /// falling back to `modelDefault`'s value; `do_sample: false` gives greedy.
    public static func generationConfig(dir: String) -> SamplingParams {
        var p = modelDefault
        guard let d = try? Data(contentsOf: URL(fileURLWithPath: dir + "/generation_config.json")),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return p }
        func f(_ k: String) -> Float? { (j[k] as? NSNumber).map { $0.floatValue } }
        if let v = f("temperature") { p.temperature = v }
        if let v = (j["top_k"] as? NSNumber)?.intValue { p.topK = v }
        if let v = f("top_p") { p.topP = v }
        if let v = f("min_p") { p.minP = v }
        if let v = f("repetition_penalty") { p.repetitionPenalty = v }
        if let v = f("presence_penalty") { p.presencePenalty = v }
        if let s = j["do_sample"] as? Bool, !s { p.temperature = 0 }
        return p
    }

    /// One line for the startup line, the settings sheet and the JSON.
    public var summary: String {
        if isPlainArgmax { return "greedy (argmax)" }
        var s = isGreedy ? "greedy (argmax of the penalised logits)" : String(format: "temperature %g, top_k %d, top_p %g, min_p %g, seed %llu",
                                                                       temperature, topK, topP, minP, seed)
        if repetitionPenalty != 1 { s += String(format: ", repetition penalty %g", repetitionPenalty) }
        if presencePenalty != 0 { s += String(format: ", presence penalty %g", presencePenalty) }
        if frequencyPenalty != 0 { s += String(format: ", frequency penalty %g", frequencyPenalty) }
        return s
    }

    public var dict: [String: Any] {
        ["temperature": Double(temperature), "top_k": topK, "top_p": Double(topP), "min_p": Double(minP),
         "presence_penalty": Double(presencePenalty), "repetition_penalty": Double(repetitionPenalty),
         "frequency_penalty": Double(frequencyPenalty), "seed": seed, "greedy": isPlainArgmax]
    }
}
