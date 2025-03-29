import React from "react";
import { Loader2 } from "lucide-react";

const LoadingAnimation = () => {
  return (
    <div className="flex flex-col items-center justify-center p-8">
      <div className="relative">
        <Loader2 className="w-12 h-12 text-purple-500 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-6 h-6 bg-gray-900 rounded-full"></div>
        </div>
      </div>
      <div className="mt-4 text-gray-400 font-medium">
        <span className="animate-pulse">Analyzing your text</span>
      </div>
      <div className="mt-2 flex space-x-1">
        <div
          className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"
          style={{ animationDelay: "0ms" }}
        ></div>
        <div
          className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"
          style={{ animationDelay: "150ms" }}
        ></div>
        <div
          className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"
          style={{ animationDelay: "300ms" }}
        ></div>
      </div>
    </div>
  );
};

export default LoadingAnimation;
