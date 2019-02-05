package com.example.maximum.cppai;

import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.drawable.AnimatedVectorDrawable;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.view.ActionMode;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    public static final String CELSIUS_KEY = "celsius", NICKNAME_KEY = "nickname";


    private boolean menuOpen = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setSupportActionBar((Toolbar)findViewById(R.id.my_toolbar));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.clear:
                Toast.makeText(this, "Cleared history", Toast.LENGTH_SHORT).show();
                return true;
            case R.id.settings:
                showSettings();
                return true;
            default:
                return super.onContextItemSelected(item);
        }
    }


    public void showSettings(){
        final AlertDialog.Builder settingsBuilder = new AlertDialog.Builder(MainActivity.this);
        View sView = getLayoutInflater().inflate(R.layout.settings_dialog,null);

        final SharedPreferences settings = getSharedPreferences("Settings",MODE_PRIVATE);

        final EditText nickname = (EditText)sView.findViewById(R.id.nickname);
        nickname.setText(settings.getString(NICKNAME_KEY,""));

        final RadioButton celsius = (RadioButton)sView.findViewById(R.id.celsius);
        RadioButton fahrenheit = (RadioButton)sView.findViewById(R.id.fahrenheit);
        if(settings.getBoolean(CELSIUS_KEY,true))
            celsius.setChecked(true);
        else
            fahrenheit.setChecked(true);


        settingsBuilder.setTitle("Settings");
        settingsBuilder.setNegativeButton("Discard", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
            }
        });
        settingsBuilder.setPositiveButton("Save", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                settings.edit().putBoolean(CELSIUS_KEY,celsius.isChecked()).putString(NICKNAME_KEY,nickname.getText().toString().trim()).apply();
                Toast.makeText(MainActivity.this, "name: "+nickname.getText().toString()+"\ntemp: "+(celsius.isChecked()?"Celsius":"fahrenheit"), Toast.LENGTH_SHORT).show();
            }
        });

        settingsBuilder.setView(sView);
        settingsBuilder.create().show();

    }
}
