import React, { useState, useRef } from 'react';
import { Upload, Activity, AlertCircle, CheckCircle, XCircle, Loader2, Camera, FileText, Stethoscope, ClipboardList } from 'lucide-react';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState("VGG19");
  const fileInputRef = useRef(null);

  const API_URL = 'http://localhost:5000';

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file');
        return;
      }
      
      setSelectedFile(file);
      setError(null);
      setPrediction(null);
      
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', selectedModel);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Prediction failed');

      const data = await response.json();
      setPrediction(data);
      
    } catch (err) {
      setError('Failed to get prediction. Make sure the backend server is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const getSeverityBadgeColor = (className) => {
    const colors = {
      'COVID': 'bg-red-600',
      'Lung_Opacity': 'bg-orange-600',
      'Viral Pneumonia': 'bg-yellow-600',
      'Normal': 'bg-green-600'
    };
    return colors[className] || 'bg-blue-600';
  };

  // Parse bullet points from text - improved version
  const parseBulletPoints = (text) => {
    if (!text) return [];
    const lines = text.split('\n').filter(line => line.trim());
    return lines
      .map(line => line.replace(/^[•\-\*]\s*/, '').trim())
      .filter(line => line.length > 0 && !line.match(/^(&\s*)?PRECAUTIONS?:?$/i))
      .filter(Boolean);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Activity className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">COVID-19 X-Ray Classifier</h1>
              <p className="text-sm text-gray-500">AI-Powered Radiography Analysis System</p>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Camera className="w-5 h-5 mr-2 text-blue-600" />
            Upload X-Ray Image
          </h2>
          
          <div 
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all"
          >
            {preview ? (
              <div className="space-y-4">
                <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-md" />
                <p className="text-sm text-gray-600">{selectedFile?.name}</p>
              </div>
            ) : (
              <div className="space-y-3">
                <Upload className="w-16 h-16 mx-auto text-gray-400" />
                <div>
                  <p className="text-lg font-medium text-gray-700">Click to upload X-ray image</p>
                  <p className="text-sm text-gray-500">PNG, JPG up to 10MB</p>
                </div>
              </div>
            )}
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Select Model
            </label>
            <select
              className="w-full border rounded-lg p-2"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="VGG19">VGG19</option>
              <option value="ResNet18">ResNet18</option>
              <option value="ViT_Small">ViT Small</option>
            </select>
          </div>

          <div className="flex space-x-3 mt-4">
            <button
              onClick={handlePredict}
              disabled={!selectedFile || loading}
              className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Analyzing Image...
                </>
              ) : (
                'Analyze X-Ray'
              )}
            </button>
            <button
              onClick={handleReset}
              disabled={!selectedFile}
              className="px-6 py-3 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Reset
            </button>
          </div>

          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
              <XCircle className="w-5 h-5 text-red-600 mr-2 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}
        </div>

        {prediction && prediction.medical_report && (
          <div className="space-y-6">
            {/* Medical Report Header */}
            <div className="bg-gradient-to-r from-slate-800 to-slate-900 rounded-xl shadow-xl p-6 text-white">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <FileText className="w-8 h-8" />
                  <div>
                    <h2 className="text-2xl font-bold">MEDICAL IMAGING ANALYSIS</h2>
                    <p className="text-sm text-gray-300 mt-1">AI-Generated Preliminary Assessment</p>
                  </div>
                </div>
                <div className={`${getSeverityBadgeColor(prediction.predicted_class)} px-4 py-2 rounded-lg`}>
                  <p className="text-xs font-semibold">CLASSIFICATION</p>
                  <p className="text-lg font-bold">{prediction.predicted_class}</p>
                </div>
              </div>
              
              <div className="border-t border-gray-600 pt-4 grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">Model</p>
                  <p className="font-semibold">{prediction.model_name}</p>
                </div>
                <div>
                  <p className="text-gray-400">Confidence</p>
                  <p className="font-semibold">{(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-gray-400">Processing Time</p>
                  <p className="font-semibold">{prediction.processing_time}ms</p>
                </div>
              </div>
            </div>

            {/* Two-Column Layout for Sections */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Section 1: What We Found */}
              <div className="bg-white rounded-xl shadow-lg border-l-4 border-purple-600 p-6 h-full">
                <div className="flex items-center space-x-2 mb-4">
                  <div className="bg-purple-100 p-2 rounded-lg">
                    <Activity className="w-5 h-5 text-purple-600" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900">What We Found</h3>
                </div>
                <div className="space-y-2.5">
                  {parseBulletPoints(prediction.medical_report.clinical_interpretation).map((point, idx) => (
                    <div key={idx} className="flex items-start space-x-2.5">
                      <span className="text-purple-600 font-bold text-lg mt-0.5">•</span>
                      <p className="text-gray-700 text-sm leading-relaxed flex-1">{point}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Section 2: Recommendations */}
              <div className="bg-white rounded-xl shadow-lg border-l-4 border-green-600 p-6 h-full">
                <div className="flex items-center space-x-2 mb-4">
                  <div className="bg-green-100 p-2 rounded-lg">
                    <ClipboardList className="w-5 h-5 text-green-600" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900">Recommendations</h3>
                </div>
                <div className="space-y-2.5">
                  {parseBulletPoints(prediction.medical_report.recommendations).map((point, idx) => (
                    <div key={idx} className="flex items-start space-x-2.5">
                      <span className="text-green-600 font-bold text-lg mt-0.5">•</span>
                      <p className="text-gray-700 text-sm leading-relaxed flex-1">{point}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Detailed Probabilities */}
            <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Probability Distribution</h3>
              <div className="space-y-3">
                {Object.entries(prediction.probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([className, prob]) => (
                    <div key={className} className="space-y-1">
                      <div className="flex justify-between items-center text-sm">
                        <span className="font-semibold text-gray-700">{className}</span>
                        <span className="font-mono text-gray-900 font-bold">{(prob * 100).toFixed(2)}%</span>
                      </div>
                      <div className="bg-gray-200 rounded-full h-2.5 overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-500 ${
                            className === prediction.predicted_class ? 'bg-blue-600' : 'bg-gray-400'
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* Medical Disclaimer */}
            <div className="bg-yellow-50 border-l-4 border-yellow-400 rounded-xl p-5">
              <div className="flex items-start">
                <AlertCircle className="w-5 h-5 text-yellow-600 mr-3 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-yellow-800">
                  <p className="font-bold mb-1">MEDICAL DISCLAIMER</p>
                  <p>
                    This tool (Powered with AI) is for research purposes only. Always consult qualified medical professionals.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;