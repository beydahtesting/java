package com.example.mymcqscannerapp;

public class AppMode {
    private static boolean onlineMode = true;  // Default mode is Online.

    public static boolean isOnlineMode() {
        return onlineMode;
    }

    public static void setOnlineMode(boolean mode) {
        onlineMode = mode;
    }
}
