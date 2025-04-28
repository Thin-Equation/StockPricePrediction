import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
});

export const uploadCSV = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload-csv/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const trainModel = async (filename: string, addIndicators: boolean, epochs: number, batchSize: number) => {
  const formData = new FormData();
  formData.append('filename', filename);
  formData.append('add_indicators', addIndicators.toString());
  formData.append('epochs', epochs.toString());
  formData.append('batch_size', batchSize.toString());
  
  const response = await api.post('/predict/', formData);
  return response.data;
};

export const forecastPrices = async (filename: string, addIndicators: boolean) => {
  const formData = new FormData();
  formData.append('filename', filename);
  formData.append('add_indicators', addIndicators.toString());
  
  const response = await api.post('/forecast/', formData);
  return response.data;
};

export const getFilesList = async () => {
  const response = await api.get('/files/');
  return response.data.files;
};

// New methods for dynamic model features

export const updateModel = async (filename: string, learningRate: number = 0.0005, epochs: number = 5) => {
  const formData = new FormData();
  formData.append('filename', filename);
  formData.append('learning_rate', learningRate.toString());
  formData.append('epochs', epochs.toString());
  
  const response = await api.post('/update-model/', formData);
  return response.data;
};

export const ensemblePredict = async (filename: string, addIndicators: boolean = true, numSamples: number = 10) => {
  const formData = new FormData();
  formData.append('filename', filename);
  formData.append('add_indicators', addIndicators.toString());
  formData.append('num_samples', numSamples.toString());
  
  const response = await api.post('/ensemble-predict/', formData);
  return response.data;
};

export const getModelInfo = async () => {
  const response = await api.get('/model-info/');
  return response.data;
};

export const getMarketRegime = async (filename: string) => {
  const response = await api.get(`/market-regime/?filename=${filename}`);
  return response.data;
};

export default api;