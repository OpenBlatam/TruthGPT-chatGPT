//this is a higher-order component that wraps other components placing them in footer

var style = {
    backgroundColor: "#F8F8F8",
    borderTop: "1px solid #E7E7E7",
    textAlign: "center",
    padding: "20px",
    position: "fixed",
    left: "0",
    bottom: "0",
    height: "60px",
    width: "100%",
};

const Footer = React.createClass({
    render: function() {
        return (
            <div style={style}>
                {this.props.children}
            </div>
        );
    }
});

export default Footer;
