import React from 'react'
import AppBar from '@material-ui/core/AppBar'
import Toolbar from '@material-ui/core/Toolbar'
import Typography from '@material-ui/core/Typography'
import UploadFiles from './components/UploadFiles';
// import Box from '@material-ui/core/Box'
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles(theme => ({
  title: {
    flexGrow: 1,
    textAlign: 'center',
  },
}));

function App() {
  const classes = useStyles();
  return (
    <div>
      <AppBar color="primary" position="static">
        <Toolbar>
          <Typography variant="title" className={classes.title} color="inherit">
              Cassava Leaf Disease Detector
          </Typography>
        </Toolbar>
      </AppBar>  
      <UploadFiles/>
    </div>
      
  );
}

export default App;
