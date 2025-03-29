import React from "react";
import { BrainCircuit } from "lucide-react";

const Header = () => {
  return (
    <header className="py-6 mb-8 border-b border-gray-800">
      <div className="flex items-center justify-center">
        <BrainCircuit className="w-8 h-8 mr-3 text-purple-500" />
        <h1 className="text-2xl font-bold text-white">
          Fraud <span className="text-purple-500">Detection & Analysis</span>
        </h1>
      </div>
      <p className="mt-2 text-center text-gray-400 max-w-md mx-auto">
        Enter text below to check for potential fraud, scams, or misinformation
        with AI-powered analysis.
      </p>
    </header>
  );
};

export default Header;
