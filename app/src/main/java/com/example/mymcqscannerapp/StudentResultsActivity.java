package com.example.mymcqscannerapp;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.File;
import java.io.FileOutputStream;
import java.util.List;

public class StudentResultsActivity extends BaseActivity {
    private TableLayout tableLayout;
    private Button btnSaveEdit, btnClearData, btnAddRecord, btnSetDefaults;
    private boolean isEditing = false; // Default mode: non-editable (saved state)

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_student_results);
        setUpToolbar();
        if(getSupportActionBar() != null){
            getSupportActionBar().setTitle("Student Records");
        }

        tableLayout = findViewById(R.id.tableLayout);
        btnSaveEdit = findViewById(R.id.btn_save_edit);
        btnClearData = findViewById(R.id.btn_clear_data);
        btnAddRecord = findViewById(R.id.btn_add_record);
        btnSetDefaults = findViewById(R.id.btn_set_defaults);

        btnSetDefaults = findViewById(R.id.btn_set_defaults);
        btnSetDefaults.setOnClickListener(v -> showSetDefaultsDialog());
        // Start in non-editable mode.
        isEditing = false;
        btnSaveEdit.setText("Edit");
        populateTable(isEditing);

        btnSaveEdit.setOnClickListener(v -> {
            if (isEditing) {
                // Save action: update records from each row (skip header at index 0)
                int childCount = tableLayout.getChildCount();
                for (int i = 1; i < childCount; i++) {
                    TableRow row = (TableRow) tableLayout.getChildAt(i);
                    EditText etName = (EditText) row.getChildAt(0);
                    EditText etRoll = (EditText) row.getChildAt(1);
                    EditText etScore = (EditText) row.getChildAt(2);
                    StudentRecord record = StudentRecord.getRecords().get(i - 1);
                    record.setName(etName.getText().toString());
                    record.setRollNumber(etRoll.getText().toString());
                    record.setScore(etScore.getText().toString());
                }
                isEditing = false;
                btnSaveEdit.setText("Edit");
                populateTable(isEditing);
                Toast.makeText(StudentResultsActivity.this, "Records saved", Toast.LENGTH_SHORT).show();
            } else {
                isEditing = true;
                btnSaveEdit.setText("Save");
                populateTable(isEditing);
            }
        });

        btnAddRecord.setOnClickListener(v -> {
            // Use the current default values
            StudentRecord.addRecord(new StudentRecord(DefaultValues.name, DefaultValues.roll, DefaultValues.score));
            isEditing = true;
            btnSaveEdit.setText("Save");
            populateTable(isEditing);
        });

        btnSetDefaults.setOnClickListener(v -> showSetDefaultsDialog());

        btnClearData.setOnClickListener(v -> {
            StudentRecord.clearRecords();
            tableLayout.removeAllViews();
            addHeaderRow(isEditing);
            Toast.makeText(StudentResultsActivity.this, "Student records cleared", Toast.LENGTH_SHORT).show();
        });
    }


    private void populateTable(boolean editable) {
        tableLayout.removeAllViews();
        addHeaderRow(editable);
        List<StudentRecord> records = StudentRecord.getRecords();
        for (int i = 0; i < records.size(); i++) {
            StudentRecord record = records.get(i);
            TableRow row = new TableRow(this);
            if (editable) {
                EditText etName = new EditText(this);
                etName.setText(record.getName());
                etName.setPadding(8, 8, 8, 8);
                row.addView(etName);

                EditText etRoll = new EditText(this);
                etRoll.setText(record.getRollNumber());
                etRoll.setPadding(8, 8, 8, 8);
                row.addView(etRoll);

                EditText etScore = new EditText(this);
                etScore.setText(record.getScore());
                etScore.setPadding(8, 8, 8, 8);
                row.addView(etScore);
            } else {
                TextView tvName = new TextView(this);
                tvName.setText(record.getName());
                tvName.setPadding(8, 8, 8, 8);
                row.addView(tvName);

                TextView tvRoll = new TextView(this);
                tvRoll.setText(record.getRollNumber());
                tvRoll.setPadding(8, 8, 8, 8);
                row.addView(tvRoll);

                TextView tvScore = new TextView(this);
                tvScore.setText(record.getScore());
                tvScore.setPadding(8, 8, 8, 8);
                row.addView(tvScore);
            }
            // Add Delete button for each row.
            Button btnDelete = new Button(this);
            btnDelete.setText("Delete");
            btnDelete.setOnClickListener(v -> {
                int index = tableLayout.indexOfChild(row) - 1; // Exclude header row.
                if (index >= 0 && index < StudentRecord.getRecords().size()) {
                    StudentRecord.getRecords().remove(index);
                    populateTable(editable);
                }
            });
            row.addView(btnDelete);
            tableLayout.addView(row);
        }
    }

    private void addHeaderRow(boolean editable) {
        TableRow header = new TableRow(this);

        TextView headerName = new TextView(this);
        headerName.setText("Student Name");
        headerName.setPadding(8, 8, 8, 8);
        headerName.setTypeface(null, android.graphics.Typeface.BOLD);
        header.addView(headerName);

        TextView headerRoll = new TextView(this);
        headerRoll.setText("Roll Number");
        headerRoll.setPadding(8, 8, 8, 8);
        headerRoll.setTypeface(null, android.graphics.Typeface.BOLD);
        header.addView(headerRoll);

        TextView headerScore = new TextView(this);
        headerScore.setText("Score");
        headerScore.setPadding(8, 8, 8, 8);
        headerScore.setTypeface(null, android.graphics.Typeface.BOLD);
        header.addView(headerScore);

        TextView headerAction = new TextView(this);
        headerAction.setText("Action");
        headerAction.setPadding(8, 8, 8, 8);
        headerAction.setTypeface(null, android.graphics.Typeface.BOLD);
        header.addView(headerAction);

        tableLayout.addView(header);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the export menu to show the download/export icon.
        getMenuInflater().inflate(R.menu.menu_export, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle export icon click.
        if (item.getItemId() == R.id.action_export) {
            showExportDialog();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void showExportDialog() {
        String[] options = {"Export as PDF", "Export as Excel"};
        new AlertDialog.Builder(this)
                .setTitle("Export Records")
                .setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        if (which == 0) {
                            exportRecords("PDF");
                        } else if (which == 1) {
                            exportRecords("Excel");
                        }
                    }
                })
                .show();
    }

    private void exportRecords(String format) {
        if (format.equals("PDF")) {
            exportRecordsAsPDF();
        } else if (format.equals("Excel")) {
            exportRecordsAsExcel();
        }
    }
    private void showSetDefaultsDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Set Default Values");

        View dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_set_defaults, null);
        EditText etDefaultName = dialogView.findViewById(R.id.et_default_name);
        EditText etDefaultRoll = dialogView.findViewById(R.id.et_default_roll);
        EditText etDefaultScore = dialogView.findViewById(R.id.et_default_score);

        // Pre-fill with current default values.
        etDefaultName.setText(DefaultValues.name);
        etDefaultRoll.setText(DefaultValues.roll);
        etDefaultScore.setText(DefaultValues.score);

        builder.setView(dialogView);
        builder.setPositiveButton("Save", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                DefaultValues.name = etDefaultName.getText().toString();
                DefaultValues.roll = etDefaultRoll.getText().toString();
                DefaultValues.score = etDefaultScore.getText().toString();
                Toast.makeText(StudentResultsActivity.this, "Default values updated", Toast.LENGTH_SHORT).show();
            }
        });
        builder.setNegativeButton("Cancel", null);
        builder.show();
    }


    private void exportRecordsAsPDF() {
        try {
            // Create the PDF file name with auto-increment if needed.
            File exportDir = getExternalFilesDir(null);
            File pdfFile = new File(exportDir, "StudentRecords.pdf");
            int counter = 2;
            while (pdfFile.exists()) {
                pdfFile = new File(exportDir, "StudentRecords" + counter + ".pdf");
                counter++;
            }

            // Using iText for PDF export.
            com.itextpdf.text.Document document = new com.itextpdf.text.Document();
            com.itextpdf.text.pdf.PdfWriter.getInstance(document, new FileOutputStream(pdfFile));
            document.open();
            document.add(new com.itextpdf.text.Paragraph("Student Records"));
            document.add(new com.itextpdf.text.Paragraph(" ")); // Blank line

            com.itextpdf.text.pdf.PdfPTable table = new com.itextpdf.text.pdf.PdfPTable(3);
            table.addCell("Student Name");
            table.addCell("Roll Number");
            table.addCell("Score");
            for (StudentRecord record : StudentRecord.getRecords()){
                table.addCell(record.getName());
                table.addCell(record.getRollNumber());
                table.addCell(record.getScore());
            }
            document.add(table);
            document.close();
            Toast.makeText(this, "Exported as PDF: " + pdfFile.getAbsolutePath(), Toast.LENGTH_LONG).show();
        } catch(Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error exporting PDF: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void exportRecordsAsExcel() {
        try {
            Workbook workbook = new XSSFWorkbook();
            Sheet sheet = workbook.createSheet("Student Records");
            Row headerRow = sheet.createRow(0);
            headerRow.createCell(0).setCellValue("Student Name");
            headerRow.createCell(1).setCellValue("Roll Number");
            headerRow.createCell(2).setCellValue("Score");
            List<StudentRecord> records = StudentRecord.getRecords();
            for (int i = 0; i < records.size(); i++){
                StudentRecord record = records.get(i);
                Row row = sheet.createRow(i + 1);
                row.createCell(0).setCellValue(record.getName());
                row.createCell(1).setCellValue(record.getRollNumber());
                row.createCell(2).setCellValue(record.getScore());
            }
            // Set fixed column widths to avoid AWT dependencies.
            sheet.setColumnWidth(0, 5000);
            sheet.setColumnWidth(1, 5000);
            sheet.setColumnWidth(2, 3000);

            // Create file name with auto-increment.
            File exportDir = getExternalFilesDir(null);
            File excelFile = new File(exportDir, "StudentRecords.xlsx");
            int counter = 2;
            while (excelFile.exists()) {
                excelFile = new File(exportDir, "StudentRecords" + counter + ".xlsx");
                counter++;
            }

            FileOutputStream fos = new FileOutputStream(excelFile);
            workbook.write(fos);
            fos.close();
            workbook.close();
            Toast.makeText(this, "Exported as Excel: " + excelFile.getAbsolutePath(), Toast.LENGTH_LONG).show();
        } catch(Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error exporting Excel: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }


    @Override
    protected Intent getForwardIntent() {
        // No forward arrow on this screen.
        return null;
    }
}
