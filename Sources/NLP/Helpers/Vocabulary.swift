//
//  File.swift
//  
//
//  Created by Riccardo Cipolleschi on 13/10/21.
//

/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Abstracts the BERT vocabulary setup to two tokenID methods and four constants.
*/

import Foundation

struct Vocabulary {
    class BundleToken {}
    
    let unkownTokenID: Int
    let paddingTokenID: Int
    let separatorTokenID: Int
    let classifyStartTokenID: Int

    private let lookupDictionary: [Substring: Int]
    
    init(from fileName: String = "vocab.txt") {
        self.lookupDictionary = Vocabulary.loadVocabulary()
        self.unkownTokenID = lookupDictionary[Constants.unkownToken.asSubstring]!
        self.paddingTokenID = lookupDictionary[Constants.paddingToken.asSubstring]!
        self.separatorTokenID = lookupDictionary[Constants.separatorToken.asSubstring]!
        self.classifyStartTokenID = lookupDictionary[Constants.classifyStartToken.asSubstring]!
    }
    
    /// Imports the model's vocabulary into a [word/wordpiece token: token ID] Dictionary.
    ///
    /// - returns: A dictionary.
    /// - Tag: LoadVocabulary
    private static func loadVocabulary(from fileName: String = "vocab.txt") -> [Substring: Int] {
        let bundle = Bundle.module
        let split = fileName.split(separator: ".").map(String.init)
        guard let url = bundle.url(forResource: split[0], withExtension: split[1]) else {
            fatalError("Vocabulary file is missing")
        }
        
        guard let rawVocabulary = try? String(contentsOf: url) else {
            fatalError("Vocabulary file has no contents.")
        }
        
        let words = rawVocabulary.split(separator: "\n")
                
        guard words.first! == "[PAD]" && words.last! == "##～" else {
            fatalError("Vocabulary file contents appear to be incorrect.")
        }

        let values = 0..<words.count
        
        let vocabulary = Dictionary(uniqueKeysWithValues: zip(words, values))
        return vocabulary
    }
    
    /// Searches for a token ID for the given word or wordpiece string.
    ///
    /// - parameters:
    ///     - string: A word or wordpiece string.
    /// - returns: A token ID.
    func tokenID(of string: String) -> Int {
        let token = Substring(string)
        return tokenID(of: token)
    }

    /// Searches for a token ID for the given word or wordpiece token.
    ///
    /// - parameters:
    ///     - string: A word or wordpiece token (Substring).
    /// - returns: A token ID.
    func tokenID(of token: Substring) -> Int {
        return self.lookupDictionary[token] ?? self.unkownTokenID
    }
}

