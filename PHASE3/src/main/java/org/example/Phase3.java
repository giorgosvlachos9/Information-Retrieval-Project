package org.example;

import com.google.common.io.Files;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.UnaryOperator;

public class Phase3 {

    private final static String READ_DIR = "IR2024\\documents.txt";
    private final static String MY_RESULTS_DIR = "IR2024\\trec_eval\\myResults.txt";
    private final static String WIKI_RESULTS_DIR = "IR2024\\trec_eval\\WikiResults.txt";
    private final static String WE_MY_RESULTS_DIR = "IR2024\\trec_eval\\myResultsWE.txt";
    private final static String WE_WIKI_RESULTS_DIR = "IR2024\\trec_eval\\WikiResultsWE.txt";
    private final static String QUERIES_DIR = "IR2024\\queries.txt";

    private static boolean WANT_WORD_EMBEDDINGS_SIMILARITY;
    private static boolean WANT_WIKI_MODEL;

    public static void main(String[] args) throws IOException {

        // --------------- --------------- DECLARE SIMILARITY AND MODEL --------------- ---------------
        WANT_WORD_EMBEDDINGS_SIMILARITY = false;
        WANT_WIKI_MODEL = false;

        String RESULTS_DIR;

        String modelPath;
        File modelFile;

        if (WANT_WIKI_MODEL) {
            modelPath = "models\\wikiModel\\model.bin";
            modelFile = new File(modelPath);
            if (WANT_WORD_EMBEDDINGS_SIMILARITY) RESULTS_DIR = WE_WIKI_RESULTS_DIR;
            else RESULTS_DIR = WIKI_RESULTS_DIR;
        } else {
            modelPath = "models\\my_word2vec.zip";
            modelFile = new File(modelPath);
            if (WANT_WORD_EMBEDDINGS_SIMILARITY) RESULTS_DIR = WE_MY_RESULTS_DIR;
            else RESULTS_DIR = MY_RESULTS_DIR;
        }

        // --------------- --------------- END OF DECLARATIONS  --------------- ---------------

        // INDEX CREATION
        String indexLocation = ("index");

        Directory directory = FSDirectory.open(Paths.get(indexLocation));

        try {
            IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());

            IndexWriter writer = new IndexWriter(directory, config);

            FieldType contentFT = new FieldType(TextField.TYPE_STORED);
            contentFT.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
            contentFT.setTokenized(true);
            contentFT.setStored(true);
            contentFT.setStoreTermVectors(true);
            contentFT.setStoreTermVectorOffsets(true);
            contentFT.setStoreTermVectorPositions(true);

            FieldType codeFT = new FieldType();
            codeFT.setIndexOptions(IndexOptions.NONE);
            codeFT.setStored(true);

            // TXT FILE EDITING FOR DOCUMENTS
            String docs_file = ReadEntireFileIntoAString(READ_DIR);
            // SPLIT THE DOCUMENTS OF THE TXT FILE
            String[] docs = txtSplitter(docs_file, "///");

            String queries_file = ReadEntireFileIntoAString(QUERIES_DIR);               // ---- QUERIES ----
            String[] queries = editQueries(txtSplitter(queries_file, "///"), "Q\\d+", "");

            // ADDING DOCS TO INDEX
            for (int i = 0; i < docs.length; i++) {
                // PARSE EACH DOCUMENT AND RETURN IT AS A DocTuple OBJECT
                DocTuple document = parseDocument(docs[i]);

                // ADD DOCUMENT
                Document doc = new Document();

                // ADD FIELDS TO DOCUMENT
                doc.add(new Field("code", document.getCode(), codeFT));
                doc.add(new Field("content", document.getText(), contentFT));

                // ADD DOCUMENT TO THE INDEX
                writer.addDocument(doc);
            }
            writer.commit();


            // INITIALIZE THE INDEX READER
            IndexReader reader = DirectoryReader.open(writer);
            String fieldName = "content";

            FieldValuesSentenceIterator fieldValuesSentenceIterator = new FieldValuesSentenceIterator(reader, fieldName);

            Word2Vec vec;

            if (modelFile.exists()) {
                System.out.println("Model already exists at: " + modelPath);
                vec = WordVectorSerializer.readWord2VecModel(modelPath);

            } else {
                if (WANT_WIKI_MODEL) {
                    System.err.println("SOMETHING WENT WRONG !!!");
                }
                vec = new Word2Vec.Builder()
                        .layerSize(50)
                        .windowSize(15)
                        .tokenizerFactory(new DefaultTokenizerFactory())
                        .iterate(fieldValuesSentenceIterator)
                        .elementsLearningAlgorithm(new SkipGram<>())
                        .seed(12345)
                        .build();

                vec.fit();

                WordVectorSerializer.writeWord2VecModel(vec, modelPath);
                System.out.println("Model saved to: " + modelPath);
            }


            try {
                IndexSearcher searcher = new IndexSearcher(reader);
                // CHECK WHICH SIMILARITY TO USE
                if (WANT_WORD_EMBEDDINGS_SIMILARITY) searcher.setSimilarity(new BM25Similarity(1.5f, 0.75f));
                else searcher.setSimilarity(new WordEmbeddingsSimilarity(vec, fieldName, WordEmbeddingsSimilarity.Smoothing.MEAN));

                String qCode = "";

                BufferedWriter fileWriter = new BufferedWriter(new FileWriter(RESULTS_DIR, StandardCharsets.UTF_8));
                for (int i = 0; i < queries.length; i++) {

                    System.out.println("Query " + i);
                    int temp = i + 1;
                    qCode = (i < 9) ? "Q0" + temp : "Q" + temp;
                    // CURRENT QUERY
                    String queryString = queries[i];

                    String[] split = queryString.split(" ");

                    INDArray denseAverageQueryVector = vec.getWordVectorsMean(Arrays.asList(split));

                    QueryParser parser = new QueryParser(fieldName, new WhitespaceAnalyzer());
                    Query query = parser.parse(queryString);

                    TopDocs hits = searcher.search(query, 50);
                    List<DocTuple> cosDocs = new ArrayList<>();

                    for (int j = 0; j < hits.scoreDocs.length; j++) {
                        ScoreDoc scoreDoc = hits.scoreDocs[j];
                        Document doc = searcher.doc(scoreDoc.doc);

                        // IF YOU WANT TO PRINT THE SCORE AND THE ID OF THE DOCUMENT
                        System.out.println("Code: " + doc.get("code") + " : " + scoreDoc.score);

                        Terms docTerms = reader.getTermVector(scoreDoc.doc, fieldName);

                        INDArray denseAverageDocumentVector = VectorizeUtils.toDenseAverageVector(docTerms, vec);
                        double cosineSim = Transforms.cosineSim(denseAverageQueryVector, denseAverageDocumentVector);
                        // IF YOU WANT TO PRINT THE COSINE_SIMILARITY_DENSE_AVG VALUE
                        System.out.println("cosineSimilarityDenseAvg=" + cosineSim);
                        String code = doc.getField("code").stringValue().trim();
                        cosDocs.add(new DocTuple(code, cosineSim));
                    }

                    //Sort the list based on cosSin in descending order
                    Collections.sort(cosDocs, new Comparator<DocTuple>() {
                        @Override
                        public int compare(DocTuple o1, DocTuple o2) {
                            // Sort in descending order (larger cosSin values first)
                            return Double.compare(o2.getCosineSim(), o1.getCosineSim());
                        }
                    });


                    // PRINT THE ORDERED DOCS
                    for (DocTuple doc : cosDocs) {
                        String docCode =  doc.getCode() ;
                        String docCosSim = Double.toString(doc.getCosineSim());

                        String line = qCode + " 0 " + docCode + " 0 " +  docCosSim + " myIRMethod";
                        fileWriter.write(line);
                        fileWriter.newLine(); // Add a newline after each line if needed
                        System.out.println("WRITE SUCCESSFULL");

                    }

                }

                fileWriter.close();

            } catch (ParseException e) {
                throw new RuntimeException(e);
            } finally {
                WordVectorSerializer.writeWord2VecModel(vec, "target/ch5w2v.zip");
                writer.deleteAll();
                writer.commit();
                writer.close();

                reader.close();
            }
        } finally {
            directory.close();
        }

        System.out.println("PROGRAM FINISHED SUCCESSFULLY !");

    }


    /**          -------------------- parseDocument function --------------------
     *
     * @param doc : The document we want to parse in String form
     * @return : A DocTuple object after successfully parsing the document
     */
    private static DocTuple parseDocument(String doc) {

        String code = doc.substring(0, 10);
        // CLEAR THE CODE FROM ALL LETTERS AND SPACES
        if (code.matches(".*[a-zA-Z\\s\\W]+.*")) {
            code = code.replaceAll("[a-zA-Z\\s]", " ");
            code = code.trim();
        }

        if (doc.contains(code)) doc = doc.replace(code,"");

        DocTuple dc = new DocTuple(code, doc.trim());

        return dc;

    }

    /**          -------------------- ReadEntireFileIntoAString function --------------------
     * Gets a file path and retuns the same file as a String object
     */
    private static String ReadEntireFileIntoAString(String file) throws FileNotFoundException {

        Scanner scanner = new Scanner(new File(file));
        scanner.useDelimiter("\\A"); //\\A stands for :start of a string
        String entireFileText = scanner.next();
        return entireFileText;
    }

    /**          -------------------- txtSplitter function --------------------
     *
     * @param txtfile : Stringified txt file
     * @param splitregex : Regular expression to perform the split on the file
     * @return : The string array after splitting the txt file
     */
    private static String[] txtSplitter(String txtfile, String splitregex) {
        return txtfile.split(splitregex);
    }

    /**          -------------------- editQueries function --------------------
     *
     * @param array : Array of queries to edit
     * @param regex : Regular expression to replace
     * @param replacement : Replacement String for the expression
     * @return : returns the array after successfully replacing the regex with the wanted String
     */
    private static String[] editQueries(String[] array, String regex, String replacement) {
        UnaryOperator<String> replaceRegex = s -> s.replaceAll(regex, replacement);
        Arrays.setAll(array, i -> replaceRegex.apply(array[i]));
        String[] queryArr = Arrays.stream(array)
                .map(String::trim)
                .toArray(String[]::new);
        return queryArr;
    }

}