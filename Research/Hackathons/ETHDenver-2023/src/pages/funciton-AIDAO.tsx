import { ethers } from 'ethers';
import MyDAOForm from './.';

type Props = {
    provider: ethers.providers.Web3Provider;
};

export default function Home({ provider }: Props) {
    return (
        <div>
            <h1>Create Decision Maker App</h1>
    <MyDAOForm provider={provider} />
    </div>
);
}

export async function getServerSideProps() {
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    return { props: { provider } };
}
