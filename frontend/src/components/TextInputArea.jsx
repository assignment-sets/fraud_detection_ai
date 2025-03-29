import React from "react";

const TextInputArea = ({ text, setText, isLoading }) => {
  return (
    <div className="w-full mb-4">
      {/* Font import in the component */}
      <link
        href="https://fonts.googleapis.com/css2?family=Comic+Sans+MS&display=swap"
        rel="stylesheet"
      />

      <label
        htmlFor="textInput"
        className="block mb-3 text-sm font-bold tracking-wide text-purple-400 uppercase"
        style={{ fontFamily: "'Comic Sans MS', cursive, sans-serif" }}
      >
        Text for Analysis
      </label>
      <div className="relative">
        <textarea
          id="textInput"
          rows="6"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={isLoading}
          className="w-full p-5 text-sm rounded-lg border-2 bg-gray-800 border-gray-700 text-white placeholder-gray-500 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200 shadow-inner"
          style={{
            resize: "none",
            fontFamily: "'Roboto Mono', monospace",
          }}
        ></textarea>

        {!text && !isLoading && (
          <div
            className="pointer-events-none absolute top-0 left-0 p-5 text-gray-500 text-sm italic opacity-70"
            style={{ fontFamily: "'Roboto Mono', monospace" }}
          >
            <span className="text-purple-400">#</span> Share any email, text, SMS, URL, or news for instant fraud detection and analysis
            <p
              className="mt-2 text-xs text-gray-600"
              style={{ fontFamily: "'Inter', sans-serif" }}
            >
              Our models will analyze your text and provide detailed insights.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TextInputArea;
