import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';

export default function HomeScreen({ navigation }) {
  return (
    <View className="flex-1 bg-gray-50 items-center justify-center p-6">
      <Text className="text-3xl font-bold text-gray-800 mb-4">Welcome Back!</Text>
      <Text className="text-center text-gray-600 mb-8">
        Ready to boost your fitness and get personalized AI recommendations?
      </Text>
      
      <View className="w-full space-y-4">
        <TouchableOpacity 
          className="bg-blue-600 py-4 rounded-xl items-center"
          onPress={() => navigation.navigate('Workout')}
        >
          <Text className="text-white font-bold text-lg">Start AI Workout Analysis</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}
