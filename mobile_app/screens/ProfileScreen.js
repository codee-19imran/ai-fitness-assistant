import React from 'react';
import { View, Text } from 'react-native';

export default function ProfileScreen() {
  return (
    <View className="flex-1 bg-gray-50 p-6">
      <Text className="text-2xl font-bold text-gray-800 mb-6">Your Fitness Profile</Text>
      
      <View className="bg-white p-4 rounded-xl shadow-sm mb-4">
        <Text className="text-lg font-semibold text-gray-700">Talent Prediction</Text>
        <Text className="text-blue-600 text-xl font-bold mt-2">Waiting for analysis...</Text>
      </View>
      
      <View className="bg-white p-4 rounded-xl shadow-sm">
        <Text className="text-lg font-semibold text-gray-700">Recommended Sport</Text>
        <Text className="text-green-600 text-xl font-bold mt-2">Complete a workout first</Text>
      </View>
    </View>
  );
}
