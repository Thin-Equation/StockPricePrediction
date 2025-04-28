'use client';

import { useState, useEffect } from 'react';
import FileUpload from '@/components/FileUpload';
import PredictionResults from '@/components/PredictionResults';
import { trainModel, forecastPrices, getFilesList } from '@/services/api';

interface FileDetails {
  filename: string;
  rows: number;
  columns: string[];
  first_date: string;
  last_date: string;
}

interface Metrics {
  mae: number;
  mse: number;
  rmse: number;
}

interface PredictionResult {
  message: string;
  filename: string;
  with_indicators: boolean;
  loss?: number;
  metrics: Metrics;
  plot: string;
  next_day_prediction?: number;
}

export default function Home() {
  const [fileDetails, setFileDetails] = useState<FileDetails | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [useIndicators, setUseIndicators] = useState<boolean>(true);
  const [epochs, setEpochs] = useState<number>(10);
  const [batchSize, setBatchSize] = useState<number>(256);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isForecasting, setIsForecasting] = useState<boolean>(false);
  const [results, setResults] = useState<PredictionResult | null>(null);
  const [modelTrained, setModelTrained] = useState<boolean>(false);

  const handleFileUploaded = (details: FileDetails) => {
    setFileDetails(details);
    setSelectedFile(details.filename);
    fetchUploadedFiles();
  };

  const fetchUploadedFiles = async () => {
    try {
      const files = await getFilesList();
      setUploadedFiles(files);
    } catch (error) {
      console.error('Error fetching files:', error);
    }
  };

  const handleTrain = async () => {
    if (!selectedFile) return;
    
    setIsTraining(true);
    setResults(null);
    
    try {
      const result = await trainModel(selectedFile, useIndicators, epochs, batchSize);
      setResults(result);
      setModelTrained(true); // Set modelTrained to true when training is successful
    } catch (error) {
      console.error('Error training model:', error);
      alert('Error training model. Check console for details.');
    } finally {
      setIsTraining(false);
    }
  };

  const handleForecast = async () => {
    if (!selectedFile) return;
    
    setIsForecasting(true);
    setResults(null);
    
    try {
      const result = await forecastPrices(selectedFile, useIndicators);
      setResults(result);
    } catch (error) {
      console.error('Error forecasting prices:', error);
      alert('Error forecasting prices. Check console for details.');
    } finally {
      setIsForecasting(false);
    }
  };

  // Load the list of files when the component mounts
  useEffect(() => {
    fetchUploadedFiles();
  }, []);

  // Reset modelTrained when selecting a new file
  useEffect(() => {
    setModelTrained(false);
  }, [selectedFile]);

  return (
    <main className="flex min-h-screen flex-col items-center p-6 bg-gray-50">
      <div className="w-full max-w-6xl">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold mb-2">Stock Price Prediction</h1>
          <p className="text-gray-600">
            Upload stock price CSV data to train prediction models and forecast future prices
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h2 className="text-xl font-semibold mb-4">Upload Data</h2>
              <FileUpload onFileUploaded={handleFileUploaded} />
              
              {fileDetails && (
                <div className="mt-4 p-4 bg-blue-50 rounded-md">
                  <h3 className="font-medium mb-2">File uploaded successfully!</h3>
                  <p className="text-sm">Name: {fileDetails.filename}</p>
                  <p className="text-sm">Rows: {fileDetails.rows}</p>
                  <p className="text-sm">Date range: {fileDetails.first_date} to {fileDetails.last_date}</p>
                </div>
              )}
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mt-6">
              <h2 className="text-xl font-semibold mb-4">Prediction Settings</h2>
              
              <div className="mb-4">
                <label className="block text-gray-700 text-sm font-bold mb-2">
                  Select File
                </label>
                <select 
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  value={selectedFile}
                  onChange={(e) => setSelectedFile(e.target.value)}
                >
                  <option value="">Select a file</option>
                  {uploadedFiles.map((file) => (
                    <option key={file} value={file}>
                      {file}
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="mb-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="form-checkbox h-5 w-5 text-blue-600"
                    checked={useIndicators}
                    onChange={(e) => setUseIndicators(e.target.checked)}
                  />
                  <span className="ml-2 text-gray-700">Use Technical Indicators</span>
                </label>
              </div>
              
              <div className="mb-4">
                <label className="block text-gray-700 text-sm font-bold mb-2">
                  Epochs: {epochs}
                </label>
                <input
                  type="range"
                  min="1"
                  max="50"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div className="mb-4">
                <label className="block text-gray-700 text-sm font-bold mb-2">
                  Batch Size: {batchSize}
                </label>
                <input
                  type="range"
                  min="32"
                  max="512"
                  step="32"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div className="flex flex-col space-y-3">
                <button
                  onClick={handleTrain}
                  disabled={!selectedFile || isTraining}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline disabled:bg-blue-300"
                >
                  {isTraining ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Training Model...
                    </span>
                  ) : (
                    'Train Model'
                  )}
                </button>
                
                <button
                  onClick={handleForecast}
                  disabled={!selectedFile || isForecasting || !modelTrained}
                  className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline disabled:bg-green-300"
                >
                  {isForecasting ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Forecasting...
                    </span>
                  ) : (
                    'Forecast Prices'
                  )}
                </button>
                {!modelTrained && selectedFile && !isTraining && (
                  <p className="text-xs text-amber-600 mt-1">
                    Please train the model first before forecasting
                  </p>
                )}
              </div>
            </div>
          </div>
          
          <div className="lg:col-span-2">
            <PredictionResults results={results} />
            
            {!results && !isForecasting && !isTraining && (
              <div className="bg-white p-8 rounded-lg shadow-sm border border-gray-200 flex flex-col items-center justify-center" style={{ minHeight: '400px' }}>
                <svg
                  className="w-16 h-16 text-gray-300 mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                <h3 className="text-xl font-medium text-gray-500">No Results Yet</h3>
                <p className="mt-2 text-gray-400 text-center">
                  Upload a CSV file and train a model or make a forecast to see results here
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
