'use client';

import Image from 'next/image';

interface Metrics {
  mae: number;
  mse: number;
  rmse: number;
}

interface PredictionResultsProps {
  results: {
    message: string;
    filename: string;
    with_indicators: boolean;
    loss?: number;
    metrics: Metrics;
    plot: string;
    next_day_prediction?: number;
  } | null;
}

const PredictionResults = ({ results }: PredictionResultsProps) => {
  if (!results) return null;

  return (
    <div className="mt-8 border border-gray-200 rounded-lg p-6 bg-white shadow-sm">
      <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
      
      <div className="grid grid-cols-1 gap-6 mb-6">
        <div>
          <h3 className="text-md font-medium mb-2">Model Performance</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded-md">
              <p className="text-sm text-gray-500">Mean Absolute Error</p>
              <p className="text-xl font-bold">{results.metrics.mae.toFixed(4)}</p>
            </div>
            <div className="bg-blue-50 p-4 rounded-md">
              <p className="text-sm text-gray-500">Mean Squared Error</p>
              <p className="text-xl font-bold">{results.metrics.mse.toFixed(4)}</p>
            </div>
            <div className="bg-blue-50 p-4 rounded-md">
              <p className="text-sm text-gray-500">Root Mean Squared Error</p>
              <p className="text-xl font-bold">{results.metrics.rmse.toFixed(4)}</p>
            </div>
          </div>
        </div>
        
        {results.next_day_prediction && (
          <div className="mt-4">
            <h3 className="text-md font-medium mb-2">Next Day Prediction</h3>
            <div className="bg-green-50 p-4 rounded-md">
              <p className="text-sm text-gray-500">Predicted Price</p>
              <p className="text-3xl font-bold">${results.next_day_prediction.toFixed(2)}</p>
            </div>
          </div>
        )}
        
        <div className="mt-4">
          <h3 className="text-md font-medium mb-2">Prediction Visualization</h3>
          <div className="bg-white border border-gray-200 rounded-lg p-2">
            {results.plot && (
              <div className="flex justify-center">
                <div style={{ position: 'relative', width: '100%', height: '400px' }}>
                  <Image 
                    src={`data:image/png;base64,${results.plot}`}
                    alt="Prediction Chart"
                    fill
                    style={{ objectFit: 'contain' }}
                    unoptimized // Required for base64 images
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="mt-4 text-sm text-gray-500">
        <p>File: {results.filename}</p>
        <p>Technical Indicators: {results.with_indicators ? 'Yes' : 'No'}</p>
      </div>
    </div>
  );
};

export default PredictionResults;