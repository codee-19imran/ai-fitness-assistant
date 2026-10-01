import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, Alert } from 'react-native';
import { Camera } from 'expo-camera';
import axios from 'axios';

// Replace this with your actual local IP address running FastAPI
const BACKEND_URL = 'http://YOUR_LOCAL_IP:8000/analyze-workout';

export default function WorkoutScreen() {
  const [hasPermission, setHasPermission] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [results, setResults] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const handleStartWorkout = () => {
    setIsRecording(true);
    setResults(null);
    
    // Simulate a workout session of 5 seconds
    setTimeout(async () => {
      setIsRecording(false);
      await sendWorkoutDataToBackend();
    }, 5000);
  };

  const sendWorkoutDataToBackend = async () => {
    try {
      // In a real app, this data would be extracted using @tensorflow-models/pose-detection
      // running on the camera frames. For this demo, we simulate the captured metrics.
      const simulatedData = {
        height: 175,
        weight: 70,
        age: 22,
        cadence: 130 + Math.random() * 10,
        speed: 2.5 + Math.random(),
        jump: 0.4 + Math.random() * 0.2,
        stability: 0.85,
        reaction_time: 0.4
      };

      const response = await axios.post(BACKEND_URL, simulatedData);
      setResults(response.data);
    } catch (error) {
      console.error(error);
      Alert.alert("Backend Error", "Could not connect to the FastAPI backend. Please check your IP.");
    }
  };

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <View className="flex-1">
      {!results ? (
        <Camera 
          className="flex-1" 
          type={Camera.Constants.Type.front}
          ref={cameraRef}
        >
          <View className="flex-1 bg-transparent justify-end p-6 pb-12">
            {!isRecording ? (
              <TouchableOpacity 
                className="bg-blue-600 p-4 rounded-xl items-center self-center w-full"
                onPress={handleStartWorkout}
              >
                <Text className="text-white font-bold text-lg">Start Analysis</Text>
              </TouchableOpacity>
            ) : (
              <View className="bg-red-600 p-4 rounded-xl items-center self-center w-full">
                <Text className="text-white font-bold text-lg">Analyzing Movement... (5s)</Text>
              </View>
            )}
          </View>
        </Camera>
      ) : (
        <View className="flex-1 bg-gray-50 p-6 justify-center">
          <Text className="text-3xl font-bold text-center mb-6">Workout Complete!</Text>
          
          <View className="bg-white p-4 rounded-xl shadow-sm mb-4">
            <Text className="text-lg font-bold text-gray-800 border-b border-gray-200 pb-2 mb-2">Traits</Text>
            <Text className="text-gray-700">Stamina: {results.traits.stamina}</Text>
            <Text className="text-gray-700">Strength: {results.traits.strength}</Text>
            <Text className="text-gray-700">Agility: {results.traits.agility}</Text>
            <Text className="text-gray-700">Balance: {results.traits.balance}</Text>
          </View>

          <View className="bg-white p-4 rounded-xl shadow-sm mb-8">
            <Text className="text-lg font-bold text-gray-800 border-b border-gray-200 pb-2 mb-2">Recommendation</Text>
            <Text className="text-blue-600 font-bold">Talent Level: {results.recommendation.talent_level}</Text>
            <Text className="text-green-600 font-bold">Best Sport: {results.recommendation.sport}</Text>
          </View>

          <TouchableOpacity 
            className="bg-blue-600 p-4 rounded-xl items-center"
            onPress={() => setResults(null)}
          >
            <Text className="text-white font-bold text-lg">Analyze Again</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}
