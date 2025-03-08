package com.example.mymcqscannerapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

public abstract class BaseActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Child activities must call setContentView() before calling setUpToolbar()
    }

    protected void setUpToolbar() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        if (toolbar != null) {
            setSupportActionBar(toolbar);
            // Enable the Up (back) button:
            if(getSupportActionBar() != null) {
                getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Default behavior: inflate forward arrow if getForwardIntent() is non-null.
        if (getForwardIntent() != null) {
            getMenuInflater().inflate(R.menu.menu_toolbar, menu);
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == android.R.id.home) {  // back/up arrow
            onBackPressed();
            return true;
        } else if (id == R.id.action_forward) { // forward arrow
            Intent forwardIntent = getForwardIntent();
            if (forwardIntent != null) {
                startActivity(forwardIntent);
            }
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    /**
     * Override in child activities if a forward arrow is needed.
     * @return An Intent to navigate forward, or null if none.
     */
    protected Intent getForwardIntent() {
        return null;
    }
}
