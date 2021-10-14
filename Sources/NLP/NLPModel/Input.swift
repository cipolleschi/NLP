//
//  File.swift
//  
//
//  Created by Riccardo Cipolleschi on 13/10/21.
//

import CoreML

struct Input {
    /// The maximum number of tokens the BERT model can process.
    static let maxTokens = 384
    
    // There are 2 sentinel tokens before the document, 1 [CLS] token and 1 [SEP] token.
    static private let documentTokenOverhead = 2
    
    // There are 3 sentinel tokens total, 1 [CLS] token and 2 [SEP] tokens.
    static private let totalTokenOverhead = 3

    var modelInput: BERTQAFP16Input?

    let question: TokenizedString
    let document: TokenizedString

    private let documentOffset: Int

    var documentRange: Range<Int> {
        return documentOffset..<documentOffset + document.tokens.count
    }
    
    var totalTokenSize: Int {
        return Self.totalTokenOverhead + document.tokens.count + question.tokens.count
    }
    
    /// - Tag: BERTInputInitializer
    init(documentString: String, questionString: String, using vocabulary: Vocabulary) {
        document = TokenizedString(documentString, using: vocabulary)
        question = TokenizedString(questionString, using: vocabulary)

        // Save the number of tokens before the document tokens for later.
        documentOffset = Self.documentTokenOverhead + question.tokens.count
        
        guard totalTokenSize < Self.maxTokens else {
            return
        }
        
        // Start the wordID array with the `classification start` token.
        var wordIDs = [vocabulary.classifyStartTokenID]
        
        // Add the question tokens and a separator.
        wordIDs += question.tokenIDs
        wordIDs += [vocabulary.separatorTokenID]
        
        // Add the document tokens and a separator.
        wordIDs += document.tokenIDs
        wordIDs += [vocabulary.separatorTokenID]
        
        // Fill the remaining token slots with padding tokens.
        let tokenIDPadding = Self.maxTokens - wordIDs.count
        wordIDs += Array(repeating: vocabulary.paddingTokenID, count: tokenIDPadding)

        guard wordIDs.count == Self.maxTokens else {
            fatalError("`wordIDs` array size isn't the right size.")
        }
        
        // Build the token types array in the same order.
        // The document tokens are type 1 and all others are type 0.
        
        // Set all of the token types before the document to 0.
        var wordTypes = Array(repeating: 0, count: documentOffset)
        
        // Set all of the document token types to 1.
        wordTypes += Array(repeating: 1, count: document.tokens.count)
        
        // Set the remaining token types to 0.
        let tokenTypePadding = Self.maxTokens - wordTypes.count
        wordTypes += Array(repeating: 0, count: tokenTypePadding)

        guard wordTypes.count == Self.maxTokens else {
            fatalError("`wordTypes` array size isn't the right size.")
        }

        // Create the MLMultiArray instances.
        let tokenIDMultiArray = try? MLMultiArray(wordIDs)
        let wordTypesMultiArray = try? MLMultiArray(wordTypes)
        
        // Unwrap the MLMultiArray optionals.
        guard let tokenIDInput = tokenIDMultiArray else {
            fatalError("Couldn't create wordID MLMultiArray input")
        }
        
        guard let tokenTypeInput = wordTypesMultiArray else {
            fatalError("Couldn't create wordType MLMultiArray input")
        }

        // Create the BERT input MLFeatureProvider.
        let modelInput = BERTQAFP16Input(wordIDs: tokenIDInput,
                                         wordTypes: tokenTypeInput)
        self.modelInput = modelInput
    }
}

/// Model Prediction Input Type

