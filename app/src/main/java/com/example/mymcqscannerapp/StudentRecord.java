package com.example.mymcqscannerapp;

import java.util.ArrayList;
import java.util.List;

public class StudentRecord {
    private String name;
    private String rollNumber;
    private String score;

    private static List<StudentRecord> recordList = new ArrayList<>();

    public StudentRecord(String name, String rollNumber, String score) {
        this.name = name;
        this.rollNumber = rollNumber;
        this.score = score;
    }

    public String getName() {
        return name;
    }

    public String getRollNumber() {
        return rollNumber;
    }

    public String getScore() {
        return score;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setRollNumber(String rollNumber) {
        this.rollNumber = rollNumber;
    }

    public void setScore(String score) {
        this.score = score;
    }

    public static void addRecord(StudentRecord record) {
        recordList.add(record);
    }

    public static List<StudentRecord> getRecords() {
        return recordList;
    }

    public static void clearRecords() {
        recordList.clear();
    }
}
