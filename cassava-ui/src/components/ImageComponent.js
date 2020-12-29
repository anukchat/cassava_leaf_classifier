import React from 'react'
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActionArea from '@material-ui/core/CardActionArea';
import CardMedia from '@material-ui/core/CardMedia';
import Box from '@material-ui/core/Box';

const useStyles = makeStyles({
    root: {
      maxWidth: 300,
    },
    media: {
      height: 300,
    },
  });

export default function ImageComponent(props) {
    const classes = useStyles();

    return (
        <Box>
            <Card className={classes.root}>
                <CardActionArea>
                    <CardMedia
                    className={classes.media}
                    image={props.imageUrl}
                    title="Cassava Leaf"
                    />
                </CardActionArea>
            </Card>      
        </Box>
    )
}
