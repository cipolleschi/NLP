/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Wrapper class for the BERT model that handles its input and output.
*/

import CoreML

public protocol NLPAlgorithm {
    func findAnswer(for question: String, in document: String) -> Substring
}

public class NLP: NLPAlgorithm {
    /// The underlying Core ML Model.
    var bertModel: BERTQAFP16
    let vocabulary: Vocabulary
    
    public init(from fileName: String = "vocab.txt") {
        do {
            self.bertModel = try BERTQAFP16(configuration: .init())
            self.vocabulary = Vocabulary(from: fileName)
        } catch {
            fatalError("Couldn't load BERT model due to: \(error.localizedDescription)")
        }
    }

    /// Finds an answer to a given question by searching a document's text.
    ///
    /// - parameters:
    ///     - question: The user's question about a document.
    ///     - document: The document text that (should) contain an answer.
    /// - returns: The answer string or an error descripton.
    /// - Tag: FindAnswerForQuestionInDocument
    public func findAnswer(for question: String, in document: String) -> Substring {
        // Prepare the input for the BERT model.
        let bertInput = Input(
            documentString: document,
            questionString: question,
            using: self.vocabulary
        )
        
        guard bertInput.totalTokenSize <= Input.maxTokens else {
            var message = "Text and question are too long"
            message += " (\(bertInput.totalTokenSize) tokens)"
            message += " for the BERT model's \(Input.maxTokens) token limit."
            return Substring(message)
        }
        
        // The MLFeatureProvider that supplies the BERT model with its input MLMultiArrays.
        let modelInput = bertInput.modelInput!
        
        // Make a prediction with the BERT model.
        guard let prediction = try? bertModel.prediction(input: modelInput) else {
            return "The BERT model is unable to make a prediction."
        }

        // Analyze the output form the BERT model.
        guard let bestLogitIndices = bestLogitsIndices(from: prediction,
                                                       in: bertInput.documentRange) else {
            return "Couldn't find a valid answer. Please try again."
        }

        // Find the indices of the original string.
        let documentTokens = bertInput.document.tokens
        let answerStart = documentTokens[bestLogitIndices.start].startIndex
        let answerEnd = documentTokens[bestLogitIndices.end].endIndex
        
        // Return the portion of the original string as the answer.
        let originalText = bertInput.document.original
        return originalText[answerStart..<answerEnd]
    }
}
