package com.example.maximum.cppai;

import android.content.Intent;
import android.os.CountDownTimer;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ProgressBar;

public class Splash extends AppCompatActivity {

    ProgressBar bar;
    private static int DELAY=3000; //change this value to change the millisecond length of the screen

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        bar = (ProgressBar)findViewById(R.id.progressBar); //wires the progress bar to the xml
        bar.setProgress(0); //makes sure the progress bar starts at 0

        new CountDownTimer(DELAY, DELAY/100) {
            @Override
            public void onTick(long millisUntilFinished) {
                bar.incrementProgressBy(1); //update the progress bar by one, if we need a
                                                // server connection we can update the user on that here
            }
            @Override
            public void onFinish() {
                startActivity(new Intent(Splash.this,MainActivity.class)); //launches the main activity
                finish(); //closes the splash screen activity so that the main activity's back button closes the app
            }
        }.start(); //start the timer
    }
}