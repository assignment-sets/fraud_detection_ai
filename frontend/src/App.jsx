import React, { useState } from "react";
import axios from "axios";
import {
  LoadingAnimation,
  ResultsDisplay,
  AnalyzeButton,
  TextInputArea,
  Header,
} from "./components";
import { AlertTriangle } from "lucide-react";

function App() {
  const [text, setText] = useState("");
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post("http://localhost:5000/analyze", {
        user_query: text,
      });

      setResults(response.data);
    } catch (err) {
      console.error("Error analyzing text:", err);
      setError(
        err.response?.data?.message ||
          "Failed to analyze text. Please try again later."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <Header />

        <div className="bg-gray-800 rounded-xl shadow-xl p-6 border border-gray-700">
          <TextInputArea text={text} setText={setText} isLoading={isLoading} />

          <div className="flex justify-center mt-4">
            <AnalyzeButton
              onClick={handleAnalyze}
              isLoading={isLoading}
              isDisabled={!text.trim()}
            />
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-900/30 border border-red-700 rounded-lg text-red-300 flex items-start">
              <AlertTriangle className="w-6 h-6 text-red-500 mr-2" />
              <p className="font-medium">{error}</p>
            </div>
          )}
        </div>

        {isLoading ? (
          <LoadingAnimation />
        ) : (
          <ResultsDisplay results={results} />
        )}
      </div>
    </div>
  );
}

export default App;
