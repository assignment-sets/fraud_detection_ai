import React from "react";
import { AlertTriangle, CheckCircle, AlertCircle, Info } from "lucide-react";

const ResultsDisplay = ({ results }) => {
  if (!results) return null;

  const getAlertType = () => {
    if (
      results.is_fraud_url ||
      results.is_fraud_email ||
      results.is_fraud_sms
    ) {
      return {
        type: "warning",
        icon: <AlertTriangle className="w-8 h-8 text-red-500" />,
        title: "Potential Risk Detected",
        bgColor: "bg-red-900/20",
        borderColor: "border-red-700",
        textColor: "text-red-300",
      };
    } else if (results.is_fake_news) {
      return {
        type: "info",
        icon: <AlertCircle className="w-8 h-8 text-yellow-500" />,
        title: "Misinformation Alert",
        bgColor: "bg-yellow-900/20",
        borderColor: "border-yellow-700",
        textColor: "text-yellow-300",
      };
    } else if (results.is_irrelevant_input) {
      return {
        type: "info",
        icon: <Info className="w-8 h-8 text-blue-500" />,
        title: "Information",
        bgColor: "bg-blue-900/20",
        borderColor: "border-blue-700",
        textColor: "text-blue-300",
      };
    } else {
      return {
        type: "success",
        icon: <CheckCircle className="w-8 h-8 text-green-500" />,
        title: "Analysis Complete",
        bgColor: "bg-green-900/20",
        borderColor: "border-green-700",
        textColor: "text-green-300",
      };
    }
  };

  const alertStyle = getAlertType();

  return (
    <div className="mt-8 w-full">
      <h2 className="text-xl font-bold mb-4 text-white border-b border-gray-700 pb-2">
        Analysis Results
      </h2>

      <div
        className={`rounded-lg p-6 shadow-lg border ${alertStyle.borderColor} ${alertStyle.bgColor}`}
      >
        <div className="flex items-start">
          <div className="mr-4">{alertStyle.icon}</div>
          <div>
            <h3 className={`text-lg font-bold mb-2 ${alertStyle.textColor}`}>
              {alertStyle.title}
            </h3>
            <div className="text-white font-medium">
              {results.final_reasoning_summary && (
                <p className="text-lg leading-relaxed">
                  {results.final_reasoning_summary.charAt(0).toUpperCase() +
                    results.final_reasoning_summary.slice(1)}
                </p>
              )}
            </div>

            <div className="mt-4 pt-4 border-t border-gray-700">
              <h4 className="text-sm font-bold text-gray-400 uppercase mb-2">
                Actions Taken
              </h4>
              <ul className="text-sm text-gray-400">
                {results.actions_taken &&
                  results.actions_taken.map((action, index) => (
                    <li key={index} className="mb-1 flex items-center">
                      <span className="mr-2">•</span> {action}
                    </li>
                  ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
