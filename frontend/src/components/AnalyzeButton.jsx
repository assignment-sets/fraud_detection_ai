import React from "react";
import { SearchIcon } from "lucide-react";

const AnalyzeButton = ({ onClick, isLoading, isDisabled }) => {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={isLoading || isDisabled}
      className={`flex items-center justify-center px-6 py-3 font-medium rounded-lg text-white transition-all duration-300 ${
        isDisabled
          ? "bg-gray-700 cursor-not-allowed opacity-60"
          : isLoading
          ? "bg-purple-700 cursor-not-allowed"
          : "bg-purple-600 hover:bg-purple-700 shadow-lg hover:shadow-purple-500/40 cursor-pointer"
      }`}
      style={{ fontFamily: "'Comic Sans MS', cursive, sans-serif" }}
    >
      {isLoading ? (
        <span className="flex items-center">
          <svg className="w-5 h-5 mr-2 animate-spin" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
          Processing...
        </span>
      ) : (
        <span className="flex items-center">
          <SearchIcon className="w-5 h-5 mr-2" />
          Analyze Text
        </span>
      )}
    </button>
  );
};

export default AnalyzeButton;
